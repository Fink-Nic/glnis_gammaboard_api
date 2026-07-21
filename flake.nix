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

        # Some upstream Python wheels call ctypes.util.find_library(), and Triton
        # directly runs /sbin/ldconfig -p.  That hard-coded FHS path is absent in
        # many Nix/remote environments.  Put this first on PYTHONPATH so both
        # the runtime wrapper and venvs used from nix develop resolve native
        # libraries from LD_LIBRARY_PATH and common driver locations instead.
        pythonSiteCustomize = pkgs.writeTextDir "sitecustomize.py" ''
          try:
              import ctypes.util as _ctypes_util
              import glob as _glob
              import os as _os
              import subprocess as _subprocess

              _DRIVER_DIRS = (
                  "/run/opengl-driver/lib",
                  "/usr/lib64",
                  "/usr/lib/x86_64-linux-gnu",
                  "/lib/x86_64-linux-gnu",
                  "/usr/local/cuda/lib64",
                  "/usr/lib/wsl/lib",
              )

              def _library_dirs():
                  seen = set()
                  for directory in [*_os.environ.get("LD_LIBRARY_PATH", "").split(":"), *_DRIVER_DIRS]:
                      if directory and directory not in seen and _os.path.isdir(directory):
                          seen.add(directory)
                          yield directory

              def _candidate_stems(name):
                  raw = str(name)
                  stems = [raw]
                  if not raw.startswith("lib"):
                      stems.append("lib" + raw)
                  return [stem if ".so" in stem else stem + ".so" for stem in stems]

              def _find_in_library_dirs(name):
                  for directory in _library_dirs():
                      for stem in _candidate_stems(name):
                          for match in sorted(_glob.glob(_os.path.join(directory, stem + "*"))):
                              if _os.path.isfile(match):
                                  return match
                  return None

              def _ldconfig_p_text():
                  lines = []
                  for directory in _library_dirs():
                      for match in sorted(_glob.glob(_os.path.join(directory, "lib*.so*"))):
                          if _os.path.isfile(match):
                              lines.append(f"\t{_os.path.basename(match)} (libc6,x86-64) => {match}")
                  return "\n".join(lines) + ("\n" if lines else "")

              if hasattr(_ctypes_util, "_findSoname_ldconfig"):
                  _ctypes_util._findSoname_ldconfig = _find_in_library_dirs

              _libcuda = _find_in_library_dirs("cuda")
              if _libcuda and "TRITON_LIBCUDA_PATH" not in _os.environ:
                  _os.environ["TRITON_LIBCUDA_PATH"] = _os.path.dirname(_libcuda)

              _original_check_output = _subprocess.check_output

              def _check_output(args, *pargs, **kwargs):
                  if list(args) == ["/sbin/ldconfig", "-p"]:
                      text = _ldconfig_p_text()
                      if kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding"):
                          return text
                      return text.encode()
                  return _original_check_output(args, *pargs, **kwargs)

              _subprocess.check_output = _check_output
          except Exception:
              pass
        '';

        ldconfigShim = pkgs.writeShellScriptBin "ldconfig" ''
          if [ "''${1:-}" = "-p" ]; then
            exit 0
          fi
          exec ${pkgs.glibc.bin}/bin/ldconfig "$@"
        '';

        pythonPathPrefix = "${pythonSiteCustomize}";
        toolPath = pkgs.lib.makeBinPath [ ldconfigShim pkgs.glibc.bin ];

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
          version = "1.5.0";
          format = "wheel";

          src = pkgs.fetchPypi {
            inherit pname version format;
            dist = "cp37";
            python = "cp37";
            abi = "abi3";
            platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
            hash = "sha256-s5iUgqsQPFdAyLhVy8avHBJmEsrHEdi1tc21Hq8dw/A=";
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

        pythonEnv = python.withPackages (ps: [
          gammaboard-process
          ps.numpy
          ps.torch-bin
          ps.setuptools
          ps.wheel
          ps.pip
          ps.tomlkit
          ps.pydot
          ps.matplotlib
          ps.scipy
          gvar
          vegas
          symbolica
          momtrop
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
export PYTHONPATH="@pythonPathPrefix@:@out@/src:''${PYTHONPATH:-}"
export LD_LIBRARY_PATH="@libPath@:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
export PATH="@toolPath@:''${PATH:-}"
export LDCONFIG="@ldconfig@"
export UV_PYTHON_DOWNLOADS=never
export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
exec @python@ "$@"
WRAPPER

            cat > $out/bin/glnis-gammaboard-sampler <<'WRAPPER'
#!/bin/sh
exec @out@/bin/python -u -m run_sampler "$@"
WRAPPER

            substituteInPlace $out/bin/python \
              --replace-fail "@out@" "$out" \
              --replace-fail "@pythonPathPrefix@" "${pythonPathPrefix}" \
              --replace-fail "@libPath@" "${libPath}" \
              --replace-fail "@toolPath@" "${toolPath}" \
              --replace-fail "@ldconfig@" "${ldconfigShim}/bin/ldconfig" \
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
            ldconfigShim
          ] ++ libs;

          shellHook = ''
            export PYTHONPATH="${pythonPathPrefix}:$PWD/src:''${PYTHONPATH:-}"
            export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
            export PATH="${toolPath}:''${PATH:-}"
            export LDCONFIG="${ldconfigShim}/bin/ldconfig"
            export UV_PYTHON_DOWNLOADS=never
            export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-64}
          '';
        };
      });
}
