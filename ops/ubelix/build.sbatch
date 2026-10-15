#!/usr/bin/env bash
#SBATCH --job-name=glnis-build
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Submit with: python3 ubelix.py [options]" >&2
    exit 2
fi

source_dir="$1"
output_path="$2"
submission_path="$3"
scratch_base="$4"
python_bin="$5"
build_cores="${SLURM_CPUS_PER_TASK:-4}"

for tool in apptainer flock; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required command not found: $tool" >&2
        exit 1
    fi
done
if [[ ! "$build_cores" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid SLURM_CPUS_PER_TASK: $build_cores" >&2
    exit 1
fi

mkdir -p "$(dirname "$output_path")" "$scratch_base/tmp" "$scratch_base/cache"
# Keep the lock file in place: deleting it would allow competing locks on
# different inodes. Each image path has its own lock.
exec 9>"${output_path}.lock"
if ! flock -n 9; then
    echo "Another build is writing $output_path" >&2
    exit 1
fi

stage_dir=""
tmp_dir=""
cleanup() {
    [[ -z "$stage_dir" ]] || rm -rf -- "$stage_dir"
    [[ -z "$tmp_dir" ]] || rm -rf -- "$tmp_dir"
    return 0
}
trap cleanup EXIT
stage_dir="$(mktemp -d "${output_path}.build-${SLURM_JOB_ID:-$$}.XXXXXX")"
tmp_dir="$(mktemp -d "$scratch_base/tmp/glnis-${SLURM_JOB_ID:-$$}.XXXXXX")"
export APPTAINER_TMPDIR="$tmp_dir"
export APPTAINER_CACHEDIR="$scratch_base/cache"
image_path="$stage_dir/glnis.sif"
started_at="$(date +%s)"

echo "source: $source_dir"
echo "output: $output_path"
echo "build cores: $build_cores"
echo "Apptainer temporary directory: $APPTAINER_TMPDIR"
echo "Apptainer cache: $APPTAINER_CACHEDIR"
apptainer --version

# %files paths in apptainer.def are relative to this directory. Nix and the
# Rust release build run inside Apptainer; neither is needed on the host.
cd "$source_dir"
apptainer build --notest --build-arg "BUILD_CORES=$build_cores" "$image_path" apptainer.def
# Run the definition's import/Rust-extension smoke test with a clean host
# environment. No GPU is needed to build or import the runtime.
apptainer test --cleanenv "$image_path"

"$python_bin" - "$submission_path" "$image_path" "$stage_dir/metadata.json" "$started_at" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

submission_path, image_path, metadata_path, started_at = sys.argv[1:]
metadata = json.loads(Path(submission_path).read_text())
image = Path(image_path)
digest = hashlib.sha256()
with image.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
metadata.update({
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "build_host": socket.gethostname(),
    "build_profile": "release",
    "build_cores": int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    "apptainer_version": subprocess.check_output(["apptainer", "--version"], text=True).strip(),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "build_seconds": int(time.time()) - int(started_at),
    "image_size_bytes": image.stat().st_size,
    "image_sha256": digest.hexdigest(),
})
Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n")
PY

# Stage on the output filesystem so publishing the image is an atomic rename.
mv -f -- "$image_path" "$output_path"
mv -f -- "$stage_dir/metadata.json" "${output_path}.meta.json"
echo "Image ready: $output_path"
echo "Build metadata: ${output_path}.meta.json"
