{
  description = "glNIS GammaBoard API development environment";

  inputs = {
    # Match GammaBoard's package set, including its Python 3.13.12 interpreter.
    nixpkgs.url = "github:NixOS/nixpkgs/b6067cc0127d4db9c26c79e4de0513e58d0c40c9";
    # Keep Maturin >= 1.12 without changing the runtime's Python package set.
    build-tools-nixpkgs.url = "github:NixOS/nixpkgs/8edc0c72e3a38faf5434e40d1f19431125fe4b30";
    fenix = {
      url = "github:nix-community/fenix/04b421dace0f5bf18d3ca363e95e59395d4cc680";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-analyzer-src.follows = "";
    };
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";

    madnis-src = {
      url = "github:madgraph-ml/madnis/main";
      flake = false;
    };
    gammaboard-src = {
      url = "github:alphal00p/gammaboard/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, build-tools-nixpkgs, fenix, flake-utils, crane, madnis-src, gammaboard-src, ... }:
    # The pinned scientific wheels below target Linux x86_64.
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        inherit (pkgs) lib;
        python = pkgs.python313;
        maturin = build-tools-nixpkgs.legacyPackages.${system}.maturin;
        projectVersion = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
        # Use this package set's matching PyTorch, Triton, and CUDA 12 stack.
        torch = python.pkgs.torch-bin.overridePythonAttrs (old: {
          dependencies = (old.dependencies or []) ++ [ python.pkgs.fsspec ];
          # The fetched filename omits the wheel's local CUDA version. Restore
          # it so wheel unpack/repack can find the matching dist-info directory.
          preBuild = (old.preBuild or "") + ''
            for torch_wheel in dist/torch-${old.version}-*.whl; do
              mv "$torch_wheel" "''${torch_wheel/torch-${old.version}-/torch-${old.version}+cu128-}"
            done
          '';
          # Nix supplies these CUDA libraries through torch-bin's buildInputs.
          pythonRemoveDeps = (old.pythonRemoveDeps or []) ++ [
            "nvidia-cuda-nvrtc-cu12" "nvidia-cuda-runtime-cu12"
            "nvidia-cuda-cupti-cu12" "nvidia-cudnn-cu12" "nvidia-cublas-cu12"
            "nvidia-cufft-cu12" "nvidia-curand-cu12" "nvidia-cusolver-cu12"
            "nvidia-cusparse-cu12" "nvidia-cusparselt-cu12" "nvidia-nccl-cu12"
            "nvidia-nvshmem-cu12" "nvidia-nvtx-cu12" "nvidia-nvjitlink-cu12"
            "nvidia-cufile-cu12"
          ];
          # Use the bindings packaged with this CUDA 12.8 stack.
          pythonRelaxDeps = (old.pythonRelaxDeps or []) ++ [ "cuda-bindings" ];
        });
        # Symbolica 3.0.1 needs newer Rust than this nixpkgs pin provides.
        rustToolchain = fenix.packages.${system}.stable.withComponents [
          "cargo" "clippy" "rust-src" "rust-std" "rustc" "rustfmt"
        ];
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        rustTarget = pkgs.stdenv.hostPlatform.rust.rustcTarget;
        cargoLinkerVar = "CARGO_TARGET_${lib.toUpper (lib.replaceStrings ["-"] ["_"] rustTarget)}_LINKER";
        cc = "${pkgs.stdenv.cc}/bin/cc";
        cxx = "${pkgs.stdenv.cc}/bin/c++";

        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          gmp
          mpfr
          libmpc
        ];

        libPath = pkgs.lib.makeLibraryPath libs;

        # Only package inputs affect the wheel; examples, test outputs and local
        # environments must not invalidate the native dependency cache.
        rustSource = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [ ./Cargo.toml ./Cargo.lock ./src/rust ];
        };
        projectSource = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            ./Cargo.toml ./Cargo.lock ./pyproject.toml ./src/rust
            (lib.fileset.fileFilter (file: file.hasExt "py") ./src)
          ];
        };

        cargoVendorDir = craneLib.vendorCargoDeps {
          cargoLock = ./Cargo.lock;
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
          pname = "glnis-gammaboard-api";
          version = projectVersion;
          src = rustSource;
          inherit cargoVendorDir;
          nativeBuildInputs = [ pkgs.m4 pkgs.pkg-config python ];
          buildInputs = libs;
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
        wheel = craneLib.mkCargoDerivation ((builtins.removeAttrs rustArgs [ "cargoExtraArgs" ]) // {
          src = projectSource;
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

        madnis = python.pkgs.buildPythonPackage {
          pname = "madnis";
          version = (builtins.fromTOML (builtins.readFile "${madnis-src}/pyproject.toml")).project.version;
          src = madnis-src;
          pyproject = true;

          nativeBuildInputs = with python.pkgs; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            torch
          ];

          dontCheckRuntimeDeps = true;
          doCheck = false;
        };

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

        gvar = python.pkgs.buildPythonPackage rec {
          pname = "gvar";
          version = "13.1.7";
          format = "wheel";

          src = pkgs.fetchPypi {
            inherit pname version format;
            dist = "cp313";
            python = "cp313";
            abi = "cp313";
            platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
            hash = "sha256-WbD6mCpjSSpeg8PPuAipuEqW/n+5CSM4EC4PJN/W62w=";
          };

          propagatedBuildInputs = with python.pkgs; [ numpy scipy ];
          doCheck = false;
        };

        vegas = python.pkgs.buildPythonPackage rec {
          pname = "vegas";
          version = "6.2.1";
          format = "wheel";

          src = pkgs.fetchPypi {
            inherit pname version format;
            dist = "cp313";
            python = "cp313";
            abi = "cp313";
            platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
            hash = "sha256-fvArhcm3gyCNOilBTOKD1avL1gmv7s0yuBEEdYQcjE4=";
          };

          propagatedBuildInputs = with python.pkgs; [ numpy gvar ];
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

        glnis-gammaboard-api = python.pkgs.buildPythonPackage {
          pname = "glnis-gammaboard-api";
          version = projectVersion;
          src = wheel;
          format = "wheel";
          sourceRoot = ".";
          unpackPhase = ''
            runHook preUnpack
            mkdir -p dist
            cp "$src"/*.whl dist/
            runHook postUnpack
          '';

          propagatedBuildInputs = with python.pkgs; [
            gammaboard-process
            numpy
            torch
            tomlkit
            pydot
            matplotlib
            scipy
            pytest
            gvar
            vegas
            symbolica
            madnis
          ];

          pythonImportsCheck = [ "glnis._rust" "glnis_runtime_bootstrap" "symbolica" "madnis_sampler" "run_sampler" ];
          # The imports-check hook has no pre-hook; prepare its environment
          # during fixup, before the Python validation phases run.
          preFixup = ''
            export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"
            # Symbolica's restricted-mode import needs a writable permit-lock
            # directory. Build checks must work without a user's license key.
            export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
            export XDG_RUNTIME_DIR="$TMPDIR/xdg-runtime"
            mkdir -p -m 700 "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR"
          '';
          # Nix supplies locked source packages in place of PEP 508 Git URLs.
          dontCheckRuntimeDeps = true;
          doCheck = false;
        };

        pythonEnv = python.withPackages (ps: [
          glnis-gammaboard-api
          ps.setuptools
          ps.wheel
          ps.pip
        ]);

        runtimeConfig = pkgs.writeText "glnis-runtime.json" (builtins.toJSON {
          # The venv's GC root must retain its base interpreter as well.
          python = "${python}/bin/python3.13";
          library_dirs = map (p: "${lib.getLib p}/lib") libs;
          tool_dirs = [ "${pkgs.stdenv.cc}/bin" ];
        });

        # Run inside nix develop. The venv keeps these exact library/compiler
        # paths for launches outside the shell, including GammaBoard workers.
        developVenv = pkgs.writeShellApplication {
          name = "glnis-dev";
          runtimeInputs = [ pkgs.uv python pkgs.coreutils pkgs.nix ];
          text = ''
            if [[ "''${1:-}" == --help ]]; then
              echo "Usage: glnis-dev [venv-directory]"
              echo "Install this checkout in editable mode; defaults to .venv."
              echo "Rerun after Rust or dependency changes. Python edits are immediate."
              exit 0
            fi
            if [[ $# -gt 1 || ! -f Cargo.toml || ! -f pyproject.toml ]]; then
              echo "Run glnis-dev [venv-directory] from the project root." >&2
              exit 2
            fi
            venv_dir="$(realpath -m "''${1:-.venv}")"
            if [[ -x "$venv_dir/bin/python" ]]; then
              "$venv_dir/bin/python" -I -S -c 'import sys; assert sys.version_info[:2] == (3, 13), "Use a Python 3.13 venv"'
            fi
            # A nixpkgs update can change glibc even within Python 3.13. Keep
            # the interpreter and Rust toolchain on the same Nix generation.
            # Refresh the venv in place, preserving its installed packages.
            if [[ ! -x "$venv_dir/bin/python" ]] || \
               [[ "$(realpath "$venv_dir/bin/python")" != "$(realpath ${python}/bin/python3.13)" ]]; then
              uv venv --allow-existing --python ${python}/bin/python3.13 "$venv_dir"
            fi
            # Update native paths before any editable-install startup hooks run.
            install -m 644 ${runtimeConfig} "$venv_dir/glnis-runtime.json"
            nix-store --add-root "$venv_dir/.glnis-native-runtime" --indirect \
              --realise ${runtimeConfig} >/dev/null
            export PYO3_PYTHON="$venv_dir/bin/python"
            uv pip install --python "$venv_dir/bin/python" --reinstall-package glnis-gammaboard-api -e .
            site_packages="$("$venv_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
            # Run after Maturin's .pth has added the editable Python sources.
            echo 'import glnis_runtime_bootstrap; glnis_runtime_bootstrap.bootstrap()' > "$site_packages/zz_glnis_runtime.pth"
            echo "Ready: source $venv_dir/bin/activate"
          '';
        };

        runtime = pkgs.runCommand "glnis-gammaboard-api-runtime" {
          nativeBuildInputs = [ pkgs.makeWrapper ];
        } ''
          mkdir -p "$out/bin"
          makeWrapper ${pythonEnv}/bin/python "$out/bin/python" \
            --prefix LD_LIBRARY_PATH : "${libPath}:/run/opengl-driver/lib:/.singularity.d/libs" \
            --prefix PATH : "${pkgs.stdenv.cc}/bin" \
            --set-default CC "${cc}" \
            --set-default CXX "${cxx}" \
            --set-default OMP_NUM_THREADS 64 \
            --set UV_PYTHON_DOWNLOADS never
          makeWrapper "$out/bin/python" "$out/bin/glnis-gammaboard-sampler" \
            --add-flags "-u -m run_sampler"
        '';
      in
      {
        packages = {
          inherit runtime wheel;
          default = runtime;
        };
        apps.default = {
          type = "app";
          program = "${runtime}/bin/glnis-gammaboard-sampler";
          meta.description = "Run the glNIS GammaBoard sampler";
        };
        checks.imports = glnis-gammaboard-api;

        devShells.default = pkgs.mkShell {
          packages = [
            python
            developVenv
            pkgs.uv
            pkgs.git
            rustToolchain
            pkgs.rust-analyzer
            maturin
            pkgs.pkg-config
            pkgs.m4
          ] ++ libs;

          CC = cc;
          CXX = cxx;
          "${cargoLinkerVar}" = cc;
          GLNIS_LIBRARY_PATH = libPath;
          GLNIS_TOOL_PATH = "${pkgs.stdenv.cc}/bin";
          UV_PYTHON_DOWNLOADS = "never";
          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:/.singularity.d/libs:''${LD_LIBRARY_PATH:-}"
            export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
          '';
        };
      });
}
