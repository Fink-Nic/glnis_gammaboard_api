#!/usr/bin/env python3
"""Submit this checkout's sampler image build to UBELIX Slurm (stdlib only)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
BUILD_INPUTS = (
    "apptainer.def", "flake.nix", "flake.lock", "pyproject.toml",
    "Cargo.toml", "Cargo.lock", "src",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", "--cpus-per-task", type=positive_int, default=4)
    parser.add_argument("--mem", default="32G", help="Slurm memory (default: 32G)")
    parser.add_argument("--time", default="06:00:00", help="walltime (default: 06:00:00)")
    parser.add_argument("--account", default="gratis")
    parser.add_argument("--partition", default="epyc2")
    parser.add_argument("--qos", help="optional Slurm QoS")
    parser.add_argument("--output", type=Path, default=Path("glnis.sif"),
                        help="image path, relative to the checkout (default: glnis.sif)")
    parser.add_argument("--scratch", type=Path,
                        default=Path("/scratch/network/users") / getpass.getuser() / "glnis-apptainer",
                        help="Apptainer scratch/cache directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the submission command without creating files or submitting")
    return parser.parse_args(argv)


def checkout_path(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def source_details() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "git_revision": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=all",
                                  "--", *BUILD_INPUTS)),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None}


def snapshot_source(destination: Path) -> str:
    destination.mkdir()
    for name in BUILD_INPUTS:
        source = ROOT / name
        target = destination / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", "*.so",
            ))
        else:
            shutil.copy2(source, target)

    digest = hashlib.sha256()
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(destination).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def submission_command(args: argparse.Namespace, build_dir: Path) -> list[str]:
    log_dir = ROOT / "logs/slurm/build"
    command = [
        "sbatch", "--parsable", "--export=ALL", "--job-name=glnis-build",
        "--nodes=1", "--ntasks=1", f"--chdir={ROOT}",
        f"--output={log_dir}/%x-%j.out", f"--error={log_dir}/%x-%j.err",
        f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}",
        f"--account={args.account}", f"--partition={args.partition}",
    ]
    if args.qos:
        command.append(f"--qos={args.qos}")
    command.extend([
        str(ROOT / "ops/ubelix/build.sbatch"), str(build_dir / "source"),
        str(checkout_path(args.output)), str(build_dir / "submission.json"),
        str(checkout_path(args.scratch)), sys.executable,
    ])
    return command


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if checkout_path(args.output).is_dir():
        raise SystemExit(f"image output is a directory: {checkout_path(args.output)}")
    for name in (*BUILD_INPUTS, "ops/ubelix/build.sbatch"):
        if not (ROOT / name).exists():
            raise SystemExit(f"missing build input: {ROOT / name}")
    if args.dry_run:
        print(shlex.join(submission_command(args, ROOT / "artifacts/ubelix/DRY_RUN")))
        return 0
    if shutil.which("sbatch") is None:
        raise SystemExit("sbatch was not found; run this helper on a UBELIX login node")

    build_root = ROOT / "artifacts/ubelix"
    build_root.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs/slurm/build").mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix="build-", dir=build_root))
    try:
        details = source_details()
        details.update({
            "source_sha256": snapshot_source(build_dir / "source"),
            "source_checkout": str(ROOT),
            "source_snapshot": str(build_dir / "source"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "resources": {name: getattr(args, name) for name in
                          ("cpus", "mem", "time", "account", "partition", "qos")},
        })
        (build_dir / "submission.json").write_text(json.dumps(details, indent=2) + "\n")
        result = subprocess.run(submission_command(args, build_dir),
                                check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        shutil.rmtree(build_dir)
        message = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise SystemExit(f"build submission failed: {message.strip()}") from error

    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise SystemExit(f"unexpected sbatch response: {result.stdout.strip()!r}; "
                         f"retained source snapshot at {build_dir}")
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    print(f"Submitted build job {job_id}")
    print(f"Image: {checkout_path(args.output)}")
    print(f"Logs: {ROOT}/logs/slurm/build/glnis-build-{job_id}.{{out,err}}")
    print(f"Source: {build_dir / 'source'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
