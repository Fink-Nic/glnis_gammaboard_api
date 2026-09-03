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

        gammaloopSource = pkgs.fetchFromGitHub {
          owner = "alphal00p";
          repo = "gammaloop";
          rev = "395610143576507503fd2c785db3ba62340f4277";
          hash = "sha256-l4tnI34aznrvGNXNJHnRLAzSXE0/4cPdXS9NE89DMuU=";
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
          version = "2.1.0";
          format = "wheel";

          src = pkgs.fetchPypi {
            inherit pname version format;
            dist = "cp37";
            python = "cp37";
            abi = "abi3";
            platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
            hash = "sha256-LU9LdlTEHYDAfa3Y6SXoykC+euUWt0QvPomdvcGIgMo=";
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
          src = ./.;
          pyproject = true;

          cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
            src = ./.;
            hash = "sha256-ahnhw8vZ36kObqVM+5izu6jkoleW0G3mDRzmRWh7VRc=";
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

          preBuild = ''
            if [ -d /build/cargo-deps-vendor/source-git-0 ] && [ ! -e /build/cargo-deps-vendor/assets ]; then
              cp -R ${gammaloopSource}/assets /build/cargo-deps-vendor/assets
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
          src = ./.;

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
            pythonEnv
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
            export PYO3_PYTHON="${pythonEnv}/bin/python"
            export UV_PYTHON_DOWNLOADS=never
            export SYMBOLICA_OEM_LICENSE="SYMBOLICA_OEM_GAMMALOOP"
            export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
          '';
        };
      });
}
