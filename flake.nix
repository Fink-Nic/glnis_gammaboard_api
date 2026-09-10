{
  description = "glNIS GammaBoard API development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    madnis-src = {
      url = "github:madgraph-ml/madnis/main";
      flake = false;
    };
    gammaboard-src = {
      url = "github:alphal00p/gammaboard/main";
      flake = false;
    };
    momtrop-src = {
      url = "github:Fink-Nic/momtrop/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, madnis-src, gammaboard-src, momtrop-src, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python313;

        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          gmp
          mpfr
          libmpc
        ];

        libPath = pkgs.lib.makeLibraryPath libs;

        projectSource = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type: let
            baseName = builtins.baseNameOf path;
          in !builtins.elem baseName [ ".git" "result" "target" ".venv" "dist" ]
            && !(pkgs.lib.hasPrefix "result-" baseName);
        };

        gammaloopSource = pkgs.fetchFromGitHub {
          owner = "alphal00p";
          repo = "gammaloop";
          rev = "6c3b1ff79c34a5e34e424ddd51dd218105b48f53";
          hash = "sha256-xNpGUHYKbvrCVz+XZ2M+2+dOy9Ez5xCimuzoHRV96aE=";
        };

        madnis = python.pkgs.buildPythonPackage {
          pname = "madnis";
          version = "main";
          src = madnis-src;
          pyproject = true;

          nativeBuildInputs = with python.pkgs; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            torch-bin
          ];

          dontCheckRuntimeDeps = true;
          doCheck = false;
        };

        gammaboard-process = python.pkgs.buildPythonPackage {
          pname = "gammaboard-process";
          version = "0.1.0";
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
          version = "2.2.0";
          format = "wheel";

          src = pkgs.fetchPypi {
            inherit pname version format;
            dist = "cp37";
            python = "cp37";
            abi = "abi3";
            platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
            hash = "sha256-h5/uDCkqfmg48Y6OmQ083AApzdU1FbBEO9u/8TvCzRI=";
          };

          doCheck = false;
        };

        momtrop = python.pkgs.buildPythonPackage {
          pname = "momtrop";
          version = "main";
          src = momtrop-src;
          pyproject = true;

          cargoDeps = pkgs.rustPlatform.importCargoLock {
            lockFile = ./momtrop-Cargo.lock;
          };

          nativeBuildInputs = with pkgs.rustPlatform; [
            cargoSetupHook
            maturinBuildHook
          ];

          postPatch = ''
            cp ${./momtrop-Cargo.lock} Cargo.lock
          '';

          doCheck = false;
        };

        glnis-gammaboard-api = python.pkgs.buildPythonPackage {
          pname = "glnis-gammaboard-api";
          version = "0.1.0";
          src = projectSource;
          pyproject = true;

          cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
            src = projectSource;
            hash = "sha256-n5TeBUMPqcXqtTWBI7QZXA8cvUGXWDV8dDXb6ZZZleQ=";
          };

          nativeBuildInputs = (with pkgs.rustPlatform; [
            cargoSetupHook
            maturinBuildHook
          ]) ++ [
            pkgs.m4
          ];

          propagatedBuildInputs = with python.pkgs; [
            gammaboard-process
            numpy
            torch-bin
            tomlkit
            pydot
            matplotlib
            scipy
            pytest
            gvar
            vegas
            symbolica
            momtrop
            madnis
          ];

          SYMBOLICA_OEM_LICENSE = "SYMBOLICA_OEM_GAMMALOOP";

          # GammaLoop embeds repository assets with paths that assume its full
          # workspace checkout. Cargo vendor flattens Git packages, so restore
          # only the asset layout required by RustEmbed/include_dir at build time.
          preBuild = ''
            api_dir="$(find /build/cargo-deps-vendor -type d -name 'gammaloop-api-*' -print -quit)"
            if [ -n "$api_dir" ]; then
              vendor_root="$(dirname "$api_dir")"
              rm -rf "$vendor_root/assets" "$vendor_root/kurvst" "$vendor_root/linnest" "$(dirname "$vendor_root")/assets"
              cp -R ${gammaloopSource}/assets "$vendor_root/assets"
              cp -R ${gammaloopSource}/assets "$(dirname "$vendor_root")/assets"
              mkdir -p "$vendor_root/kurvst"
              cp -R ${gammaloopSource}/crates/kurvst/typst "$vendor_root/kurvst/typst"
              mkdir -p "$vendor_root/linnest"
              cp -R ${gammaloopSource}/crates/linnest/typst "$vendor_root/linnest/typst"
            fi
          '';

          dontCheckRuntimeDeps = true;
          doCheck = false;
        };

        pythonEnv = python.withPackages (ps: [
          glnis-gammaboard-api
          ps.setuptools
          ps.wheel
          ps.pip
        ]);

        runtime = pkgs.stdenv.mkDerivation {
          name = "glnis-gammaboard-api-runtime";
          src = projectSource;

          dontBuild = true;

          installPhase = ''
            mkdir -p $out/bin

            cat > $out/bin/python <<'WRAPPER'
#!/bin/sh
export LD_LIBRARY_PATH="@libPath@:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
export UV_PYTHON_DOWNLOADS=never
export SYMBOLICA_OEM_LICENSE="''${SYMBOLICA_OEM_LICENSE:-SYMBOLICA_OEM_GAMMALOOP}"
export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
exec @python@ "$@"
WRAPPER

            cat > $out/bin/glnis-gammaboard-sampler <<'WRAPPER'
#!/bin/sh
exec @out@/bin/python -u -m run_sampler "$@"
WRAPPER

            substituteInPlace $out/bin/python \
              --replace-fail "@libPath@" "${libPath}" \
              --replace-fail "@python@" "${pythonEnv}/bin/python"

            substituteInPlace $out/bin/glnis-gammaboard-sampler \
              --replace-fail "@out@" "$out"

            chmod +x $out/bin/python $out/bin/glnis-gammaboard-sampler
          '';
        };
      in
      {
        packages.runtime = runtime;
        packages.default = runtime;

        devShells.default = pkgs.mkShell {
          packages = [
            runtime
            pkgs.uv
            pkgs.git
            pkgs.cargo
            pkgs.clippy
            pkgs.rustc
            pkgs.rustfmt
            pkgs.rust-analyzer
            pkgs.maturin
            pkgs.pkg-config
          ] ++ libs;

          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
            export PYO3_PYTHON="${runtime}/bin/python"
            export UV_PYTHON_DOWNLOADS=never
            export SYMBOLICA_OEM_LICENSE="''${SYMBOLICA_OEM_LICENSE:-SYMBOLICA_OEM_GAMMALOOP}"
            export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
          '';
        };
      });
}
