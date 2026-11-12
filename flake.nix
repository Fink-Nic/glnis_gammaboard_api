{
  description = "glNIS GammaBoard API development environment";

  inputs = {
    # One package set supplies Python 3.13, PyTorch, Rust >=1.96 and Maturin >=1.12.
    nixpkgs.url = "github:NixOS/nixpkgs/8edc0c72e3a38faf5434e40d1f19431125fe4b30";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";

    gammaboard-src = {
      url = "github:alphal00p/gammaboard/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, crane, gammaboard-src, ... }:
    # The pinned scientific wheels below target Linux x86_64.
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        inherit (pkgs) lib;
        python = pkgs.python313;
        maturin = pkgs.maturin;
        projectVersion = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
        # The wheel in this pin targets CUDA 13.0; nixpkgs still defaults to 12.
        cudaPackages = pkgs.cudaPackages_13_0.overrideScope (_final: previous: {
          # Deploy the library, without compiling its GPU test/example programs.
          libnvshmem = previous.libnvshmem.overrideAttrs (old: {
            cmakeFlags = old.cmakeFlags ++ [
              (lib.cmakeBool "NVSHMEM_BUILD_TESTS" false)
              (lib.cmakeBool "NVSHMEM_BUILD_EXAMPLES" false)
            ];
          });
        });
        torch = python.pkgs.torch-bin.override { inherit cudaPackages; };
        craneLib = crane.mkLib pkgs;
        rustTarget = pkgs.stdenv.hostPlatform.rust.rustcTarget;
        cargoLinkerVar = "CARGO_TARGET_${lib.toUpper (lib.replaceStrings ["-"] ["_"] rustTarget)}_LINKER";
        cc = "${pkgs.stdenv.cc}/bin/cc";

        # Torch/Triton need the C/C++ runtime and compiler even in the core.
        coreLibs = with pkgs; [ stdenv.cc.cc.lib zlib ];
        gammaloopLibs = coreLibs ++ (with pkgs; [ gmp mpfr libmpc ]);
        coreLibPath = lib.makeLibraryPath coreLibs;
        gammaloopLibPath = lib.makeLibraryPath gammaloopLibs;

        # The Python core has no dependency on the native source or toolchain.
        projectSource = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            ./pyproject.toml ./README.md ./MANIFEST.in
            (lib.fileset.fileFilter (file: file.hasExt "py") ./src)
          ];
        };
        pythonBuild = python.withPackages (ps: [ ps.build ps.setuptools ps.wheel ]);
        wheel = pkgs.stdenvNoCC.mkDerivation {
          pname = "glnis-gammaboard-api-wheel";
          version = projectVersion;
          src = projectSource;
          nativeBuildInputs = [ pythonBuild ];
          buildPhase = "python -m build --wheel --no-isolation";
          installPhase = ''
            mkdir -p "$out"
            cp dist/*.whl "$out/"
          '';
        };

        # Only native package inputs affect the expensive dependency cache.
        rustSource = lib.fileset.toSource {
          root = ./native/gammaloop;
          fileset = lib.fileset.unions [
            ./native/gammaloop/Cargo.toml ./native/gammaloop/Cargo.lock
            ./native/gammaloop/src
          ];
        };
        nativeSource = lib.fileset.toSource {
          root = ./native/gammaloop;
          fileset = lib.fileset.unions [
            ./native/gammaloop/Cargo.toml ./native/gammaloop/Cargo.lock
            ./native/gammaloop/pyproject.toml ./native/gammaloop/src
            (lib.fileset.fileFilter (file: file.hasExt "py") ./native/gammaloop/python)
          ];
        };

        cargoVendorDir = craneLib.vendorCargoDeps {
          cargoLock = ./native/gammaloop/Cargo.lock;
          # GammaLoop embeds assets outside its individual crate directories.
          # Point vendored sources at the full checkout Crane already fetched
          # from Cargo.lock, so assets cannot drift to another revision.
          overrideVendorGitCheckout = packages: drv:
            if lib.any (p: lib.hasPrefix "git+https://github.com/alphal00p/gammaloop.git" p.source) packages
            then drv.overrideAttrs (old: {
              postInstall = (old.postInstall or "") + ''
                find "$out" -type f -name '*.rs' -exec sed -i -E \
                  -e 's|"(\.\./)+assets/|"${old.src}/assets/|g' \
                  -e 's|\$CARGO_MANIFEST_DIR/(\.\./)+assets/|${old.src}/assets/|g' \
                  -e 's|"\.\./linnest/build_assets.rs"|"${old.src}/crates/linnest/build_assets.rs"|g' \
                  -e 's|\$CARGO_MANIFEST_DIR/\.\./linnest/typst|${old.src}/crates/linnest/typst|g' \
                  -e 's|\$CARGO_MANIFEST_DIR/\.\./kurvst/typst|${old.src}/crates/kurvst/typst|g' {} +
                find "$out" -path '*/gammaloop-api-*/build.rs' -exec sed -i \
                  -e 's|&manifest.join("\.\.")|\&manifest.join("${old.src}/crates")|g' {} +
              '';
            }) else drv;
        };

        rustArgs = {
          pname = "glnis-gammaloop-native";
          version = projectVersion;
          src = rustSource;
          inherit cargoVendorDir;
          nativeBuildInputs = [ pkgs.m4 pkgs.pkg-config python pkgs.writableTmpDirAsHomeHook ];
          buildInputs = gammaloopLibs;
          PYO3_PYTHON = "${python}/bin/python3.13";
          # Match Maturin's environment when preparing the dependency cache.
          PYO3_ENVIRONMENT_SIGNATURE = "cpython-${python.pythonVersion}-64bit";
          PYO3_BUILD_EXTENSION_MODULE = "1";
          doCheck = false;
          cargoExtraArgs = "--locked --lib";
        };
        cargoArtifacts = craneLib.buildDepsOnly (rustArgs // {
          buildPhaseCargoCommand = "cargo build --release --locked --lib";
        });
        wheel-gammaloop = craneLib.mkCargoDerivation ((builtins.removeAttrs rustArgs [ "cargoExtraArgs" ]) // {
          src = nativeSource;
          inherit cargoArtifacts;
          nativeBuildInputs = rustArgs.nativeBuildInputs ++ [ maturin ];
          buildPhaseCargoCommand = ''
            maturin build --release --locked --offline --manylinux off \
              --interpreter ${python}/bin/python3.13 --out dist
          '';
          installPhaseCommand = ''
            mkdir -p "$out"
            cp dist/*.whl "$out/"
          '';
          doInstallCargoArtifacts = false;
        });

        gammaboard-process = python.pkgs.buildPythonPackage {
          pname = "gammaboard-process";
          version = (builtins.fromTOML (builtins.readFile "${gammaboard-src}/process_api/python/pyproject.toml")).project.version;
          src = "${gammaboard-src}/process_api/python";
          pyproject = true;

          nativeBuildInputs = with python.pkgs; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
          ];

          doCheck = false;
        };

        symbolica = python.pkgs.buildPythonPackage rec {
          pname = "symbolica";
          version = "3.0.0";
          format = "wheel";

          src = pkgs.fetchPypi {
            inherit pname version format;
            dist = "cp39";
            python = "cp39";
            abi = "abi3";
            platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
            hash = "sha256-opd/eNj0DOMga6SYUM919HN4WjKpNo5hFdBZdVpI0V4=";
          };

          doCheck = false;
        };

        mkWheelPackage = attrs: python.pkgs.buildPythonPackage ({
          version = projectVersion;
          nativeBuildInputs = [ pkgs.writableTmpDirAsHomeHook ];
          format = "wheel";
          sourceRoot = ".";
          unpackPhase = ''
            runHook preUnpack
            mkdir -p dist
            cp "$src"/*.whl dist/
            runHook postUnpack
          '';
          # Nix supplies locked source packages in place of PEP 508 Git URLs.
          dontCheckRuntimeDeps = true;
          doCheck = false;
        } // attrs);
        glnis-gammaboard-api = mkWheelPackage {
          pname = "glnis-gammaboard-api";
          src = wheel;
          propagatedBuildInputs = with python.pkgs; [
            gammaboard-process
            numpy
            torch
            matplotlib
          ];

          pythonImportsCheck = [ "glnis.core.parser" "gammaboard_sampler" "run_sampler" ];
          preFixup = ''
            export LD_LIBRARY_PATH="${coreLibPath}:''${LD_LIBRARY_PATH:-}"
          '';
        };
        glnis-gammaloop-native = mkWheelPackage {
          pname = "glnis-gammaloop-native";
          src = wheel-gammaloop;
          pythonImportsCheck = [ "glnis_gammaloop_native._rust" ];
          # The imports-check hook has no pre-hook; prepare its environment
          # during fixup, before the Python validation phases run.
          preFixup = ''
            export LD_LIBRARY_PATH="${gammaloopLibPath}:''${LD_LIBRARY_PATH:-}"
            # Symbolica's restricted-mode import needs a writable permit-lock
            # directory. Build checks must work without a user's license key.
            export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
            export XDG_RUNTIME_DIR="$TMPDIR/xdg-runtime"
            mkdir -p -m 700 "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR"
          '';
        };

        mkPythonEnv = withGammaloop: python.withPackages (ps: [
          glnis-gammaboard-api
        ] ++ lib.optionals withGammaloop [ glnis-gammaloop-native symbolica ps.pydot ]);

        # Nix wheel environments need shared libraries before Python starts.
        # Normal Python/Apptainer installs use their platform loader unchanged.
        runtimeEnvironment = withGammaloop: ''
          export LD_LIBRARY_PATH="${if withGammaloop then gammaloopLibPath else coreLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export PATH="${lib.makeBinPath [ pkgs.stdenv.cc pkgs.openssl ]}:$PATH"
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
            export TRITON_LIBCUDA_PATH="''${TRITON_LIBCUDA_PATH:-/run/opengl-driver/lib}"
          fi
        '';
        # Preserve the invoked venv path for pyvenv.cfg and generated shebangs.
        # The launcher is also a GC root for its interpreter/library/compiler paths.
        venvPython = withGammaloop: pkgs.writeShellScript "glnis-venv-python" ''
          ${runtimeEnvironment withGammaloop}
          exec -a "$0" ${python}/bin/python3.13 "$@"
        '';
        mkDevelopVenv = withGammaloop: pkgs.writeShellApplication {
          name = "glnis-dev";
          runtimeInputs = [ pkgs.uv pkgs.coreutils pkgs.nix ];
          text = ''
            if [[ "''${1:-}" == --help ]]; then
              echo "Usage: glnis-dev [venv-directory]"
              echo "Use nix develop .#gammaloop for the optional native backend."
              exit 0
            fi
            if [[ $# -gt 1 || "''${1:-}" == --* || ! -f pyproject.toml || ! -d src/glnis ]]; then
              echo "Run glnis-dev [venv-directory] from the project root." >&2
              exit 2
            fi
            venv_dir="$(realpath -m "''${1:-.venv}")"
            uv venv --allow-existing --python ${python}/bin/python3.13 "$venv_dir"
            ln -sf ${venvPython withGammaloop} "$venv_dir/bin/python"
            nix-store --add-root "$venv_dir/.nix-runtime" --indirect \
              --realise ${venvPython withGammaloop} >/dev/null
            ${if withGammaloop then ''
              uv pip install --python "$venv_dir/bin/python" -e '.[gammaloop,test]' -e ./native/gammaloop
            '' else ''
              uv pip install --python "$venv_dir/bin/python" -e '.[test]'
            ''}
            echo "Ready: source $venv_dir/bin/activate"
          '';
        };
        entryPoints = {
          glnis-gammaboard-sampler = "run_sampler";
          glnis-integrate = "glnis.scripts.run";
          glnis-compare-samplers = "glnis.scripts.sampler_comparison";
          glnis-slice-plot = "glnis.scripts.slice_plots";
        };
        mkRuntime = withGammaloop:
          let pythonEnv = mkPythonEnv withGammaloop;
          in pkgs.runCommand "glnis-runtime${lib.optionalString withGammaloop "-gammaloop"}" {} ''
            mkdir -p "$out/bin"
            cat > "$out/bin/python" <<'EOF'
            #!${pkgs.bash}/bin/bash
            ${runtimeEnvironment withGammaloop}
            exec ${pythonEnv}/bin/python "$@"
            EOF
            chmod +x "$out/bin/python"
            ${lib.concatStrings (lib.mapAttrsToList (name: module: ''
              cat > "$out/bin/${name}" <<EOF
              #!${pkgs.bash}/bin/bash
              exec "$out/bin/python" -u -m ${module} "\$@"
              EOF
              chmod +x "$out/bin/${name}"
            '') entryPoints)}
          '';
        runtime = mkRuntime false;
        runtime-gammaloop = mkRuntime true;
        mkDevShell = withGammaloop: pkgs.mkShell ({
          packages = [ python (mkDevelopVenv withGammaloop) pkgs.uv pkgs.git pkgs.stdenv.cc ]
            ++ lib.optionals withGammaloop [
              pkgs.cargo pkgs.rustc pkgs.rustfmt pkgs.rust-analyzer maturin pkgs.pkg-config pkgs.m4
            ];
          shellHook = runtimeEnvironment withGammaloop;
        } // lib.optionalAttrs withGammaloop { "${cargoLinkerVar}" = cc; });
      in
      {
        packages = {
          inherit runtime wheel runtime-gammaloop wheel-gammaloop;
          default = runtime;
        };
        apps.default = {
          type = "app";
          program = "${runtime}/bin/glnis-gammaboard-sampler";
          meta.description = "Run the glNIS GammaBoard sampler";
        };
        apps.gammaloop = {
          type = "app";
          program = "${runtime-gammaloop}/bin/glnis-gammaboard-sampler";
          meta.description = "Run the glNIS sampler with GammaLoop support";
        };
        checks = {
          imports = glnis-gammaboard-api;
          imports-gammaloop = glnis-gammaloop-native;
        };
        devShells = {
          default = mkDevShell false;
          gammaloop = mkDevShell true;
        };
      });
}
