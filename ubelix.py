#!/usr/bin/env python3
"""Freeze this checkout and submit a plain Apptainer image build to Slurm."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent
INPUTS = ("apptainer.def", "pyproject.toml", "README.md", "MANIFEST.in", "src", "ops/check_runtime.py",
          "native/gammaloop/Cargo.toml", "native/gammaloop/Cargo.lock", "native/gammaloop/pyproject.toml",
          "native/gammaloop/src", "native/gammaloop/python")


def write_record(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def run_build(path):
    # This private entry point is invoked by sbatch, never by submission itself.
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("image builds must run inside a Slurm allocation")
    path = path.resolve()
    record = json.loads(path.read_text())
    output = Path(record["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    record.update(state="running", job_id=os.environ["SLURM_JOB_ID"])
    write_record(path, record)
    staged = output.with_name(output.name + ".partial-" + record["job_id"])
    try:
        with output.with_name(output.name + ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Like GammaBoard's builder, unpack and assemble on job-local scratch;
            # retain only the source archive, record, downloads and SIF on shared storage.
            with tempfile.TemporaryDirectory(prefix="glnis-build-", dir=os.environ.get("TMPDIR")) as directory:
                scratch = Path(directory)
                with tarfile.open(path.parent / "source.tar") as archive:
                    archive.extractall(scratch / "source", filter="data")
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("APPTAINERENV_", "SINGULARITYENV_"))}
                env["APPTAINER_TMPDIR"] = str(scratch / "tmp")
                env["APPTAINER_CACHEDIR"] = record["cache"]
                Path(env["APPTAINER_TMPDIR"]).mkdir()
                Path(env["APPTAINER_CACHEDIR"]).mkdir(parents=True, exist_ok=True)
                image = scratch / "glnis.sif"
                print(f"Build scratch: {scratch}", flush=True)
                command = ["apptainer", "build", "--notest", "--mksquashfs-args",
                           f"-processors {record['cpus']}", "--build-arg", f"BUILD_CORES={record['cpus']}",
                           "--build-arg", f"WITH_GAMMALOOP={int(record['with_gammaloop'])}",
                           "--build-arg", f"TORCH_VERSION={record['torch_version']}",
                           "--build-arg", f"TORCH_INDEX={record['torch_index']}", str(image), "apptainer.def"]
                subprocess.run(command, cwd=scratch / "source", env=env, check=True)
                test_home = scratch / "home"
                test_home.mkdir()
                subprocess.run(["apptainer", "test", "--cleanenv", "--home", f"{test_home}:/home/glnis",
                                "--bind", f"{scratch / 'tmp'}:/tmp", str(image)], cwd=scratch, env=env, check=True)
                shutil.copyfile(image, staged)
                with staged.open("rb") as stream:
                    record["image_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                os.replace(staged, output)
        record.update(state="completed", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(state="failed", error=str(error))
        raise
    finally:
        staged.unlink(missing_ok=True)
        write_record(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--account", default="gratis")
    parser.add_argument("--partition", default="epyc2")
    parser.add_argument("--qos")
    parser.add_argument("--output", type=Path, default=ROOT / "glnis.sif")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/ubelix/downloads")
    parser.add_argument("--with-gammaloop", action="store_true")
    parser.add_argument("--torch-version", default="2.14.1")
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu126")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-build", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info < (3, 11, 8):
        parser.error("use Python 3.11.8 or newer (e.g. module load Python)")
    if args.run_build:
        run_build(args.run_build)
        return
    if args.cpus < 1:
        parser.error("cpus must be positive")
    output = (ROOT / args.output.expanduser()).resolve()
    if output.is_dir():
        parser.error("output must be an image file")
    logs = ROOT / "logs/slurm/build"
    build = ROOT / "artifacts/ubelix/DRY_RUN"
    command = ["sbatch", "--parsable", "--export=ALL", "--nodes=1", "--ntasks=1", "--job-name=glnis-build",
               f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}",
               f"--account={args.account}", f"--partition={args.partition}",
               f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"]
    if args.qos:
        command.append(f"--qos={args.qos}")
    if not args.dry_run:
        if shutil.which("sbatch") is None:
            parser.error("sbatch not found; submit from a UBELIX login node")
        for name in INPUTS:
            if not (ROOT / name).exists():
                parser.error(f"missing build input: {name}")
        logs.mkdir(parents=True, exist_ok=True)
        build.parent.mkdir(parents=True, exist_ok=True)
        build = Path(tempfile.mkdtemp(prefix="build-", dir=build.parent))
        shutil.copy2(__file__, build / "build.py")
        with tarfile.open(build / "source.tar", "w") as archive:
            for name in INPUTS:
                source = ROOT / name
                paths = sorted(source.rglob("*")) if source.is_dir() else [source]
                for item in paths:
                    if item.is_file() and item.suffix not in (".pyc", ".pyo", ".so", ".pyd", ".dll", ".a", ".rlib") and "__pycache__" not in item.parts:
                        archive.add(item, arcname=item.relative_to(ROOT), recursive=False)
        with (build / "source.tar").open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        write_record(build / "build.json", {
            "state": "submitted", "source_sha256": source_hash,
            "output": str(output), "cache": str((ROOT / args.cache.expanduser()).resolve()),
            "cpus": args.cpus, "with_gammaloop": args.with_gammaloop,
            "torch_version": args.torch_version, "torch_index": args.torch_index,
            "submitted_at": datetime.now(timezone.utc).isoformat()})
    command += ["--wrap", shlex.join([sys.executable, str(build / "build.py"), "--run-build", str(build / "build.json")])]
    if args.dry_run:
        print(shlex.join(command))
        return
    try:
        submitted = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        record_path = build / "build.json"
        record = json.loads(record_path.read_text())
        record.update(state="submission_failed", error=str(error))
        write_record(record_path, record)
        raise
    identifier = submitted.stdout.strip().split(";", 1)[0]
    if not identifier.isdigit():
        raise RuntimeError(f"unexpected sbatch response: {submitted.stdout!r}; inspect {build}")
    print(f"Submitted job {identifier}\nImage: {output}\nRecord: {build / 'build.json'}\nLogs: {logs}/glnis-build-{identifier}.{{out,err}}")


if __name__ == "__main__":
    main()
