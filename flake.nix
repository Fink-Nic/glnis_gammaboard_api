{
  description = "madnis gammaboard API runtime";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Prefer pinning this to a commit once it works.
    madnis-src = {
      url = "github:madgraph-ml/madnis/main";
      flake = false;
    };
    gammaboard-src = {
      url = "github:alphal00p/gammaboard/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, madnis-src, gammaboard-src, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python312;

        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          gmp
          mpfr
          libmpc
        ];

        libPath = pkgs.lib.makeLibraryPath libs;

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

        pythonEnv = python.withPackages (ps: [
          gammaboard-process
          ps.numpy
          ps.torch-bin
          ps.setuptools
          madnis
        ]);

        runtime = pkgs.stdenv.mkDerivation {
          name = "madnis-gammaboard-api-runtime";
          src = ./.;

          dontBuild = true;

          installPhase = ''
            mkdir -p $out/src $out/bin
            cp -r src/* $out/src/

            cat > $out/bin/python <<'WRAPPER'
#!/bin/sh
export PYTHONPATH="@out@/src:''${PYTHONPATH:-}"
export LD_LIBRARY_PATH="@libPath@:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
exec @python@ "$@"
WRAPPER

            substituteInPlace $out/bin/python \
              --replace-fail "@out@" "$out" \
              --replace-fail "@libPath@" "${libPath}" \
              --replace-fail "@python@" "${pythonEnv}/bin/python"

            chmod +x $out/bin/python
          '';
        };
      in
      {
        packages.runtime = runtime;
        packages.default = runtime;

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
          ] ++ libs;

          shellHook = ''
            export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
            export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
            export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
          '';
        };
      });
}
