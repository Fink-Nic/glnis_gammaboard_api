# Build and run the sampler on UBELIX

The image installs the Python sampler and prebuilt CUDA-enabled PyTorch wheels
on a standard Python 3.13 / Debian Bookworm base. There is no Nix installation,
CUDA toolkit build, driver installation, or custom library-discovery code in the
container. The default image is core-only. GammaBoard runs its evaluator in its
own environment; that does not require a GammaLoop backend in this sampler image.

Use the optional native backend only when the sampler itself reads GammaLoop
states for coordinate mappings. `--with-gammaloop` adds the Rust companion,
Symbolica and Pydot. It can take substantially longer to build.

## Submit from the login node

Clone the sampler next to GammaBoard on shared storage:

```bash
cd /storage/research/itp_localunitaritydata/<your-parent-directory>
git clone https://github.com/Fink-Nic/glnis_gammaboard_api.git
cd glnis_gammaboard_api
module load Python
python3 --version  # Python 3.11.8 or newer for the submission helper
python3 ubelix.py --dry-run
python3 ubelix.py
```

The login node only freezes source files and calls `sbatch`. Image creation is
performed by the batch job on a CPU compute node. As in GammaBoard's
`ops/ubelix/gbops/builds.py`, the queued build uses an immutable source archive
and a frozen copy of the build helper; edits and branch changes after submission
do not affect it. Unpacking and container assembly use the job's `TMPDIR`, or
local `/tmp` when unset. Only source archives, records, downloads and finished
images are stored alongside the shared checkout.

The helper needs only the Python standard library and `sbatch`. Apptainer must
be available on the compute node; UBELIX normally provides it without loading a
module. No GPU allocation is needed to build or perform the CPU smoke test.
Compute nodes need access to Docker Hub, PyPI, the PyTorch wheel index and GitHub.
See the official [container guide](https://hpc-unibe-ch.github.io/runjobs/scheduled-jobs/container-jobs/).

Defaults are account `gratis`, partition `epyc2`, four CPUs, 32 GB RAM and two
hours. Override the allocation or output as needed:

```bash
python3 ubelix.py --cpus 8 --mem 64G --time 04:00:00
python3 ubelix.py --account <account> --partition <partition> --qos <qos>
python3 ubelix.py --with-gammaloop --time 06:00:00 --output glnis-gammaloop.sif
```

Relative output/cache paths resolve against the checkout. `--cache` changes the
Apptainer download cache; it caches base-image downloads, not pip or Rust build
artifacts. `python3 ubelix.py --help` lists the options. Source snapshots include
local edits but exclude compiled Python/native artifacts. They do not include
Nix files, a virtual environment, generated physics states, or credentials.

## Monitor and publish

Submission prints the job ID and exact build-record path. For example:

```bash
squeue -j <job-id>
tail -f logs/slurm/build/glnis-build-<job-id>.out
tail -f logs/slurm/build/glnis-build-<job-id>.err
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat artifacts/ubelix/build-<identifier>/build.json
```

The record moves from `submitted` to `running`, then `completed` or `failed`.
Slurm remains authoritative for jobs killed before Python can update the record.
The build log prints its scratch directory and normal apt/pip/compiler output.

The job runs `apptainer test --cleanenv` with a temporary writable home and `/tmp`.
Its smoke test trains a small six-dimensional mixed proposal, checks densities,
and restores a checkpoint. Native builds also check the companion import.
An image is published only after these checks succeed: a temporary copy on the
output filesystem is atomically renamed to `glnis.sif`. Failed builds/tests leave
an existing image untouched. An output lock rejects concurrent builds of the
same image. The record retains the source SHA-256, selected PyTorch profile and
final image SHA-256. Installed package versions are recorded inside the image at
`/usr/local/share/glnis/packages.txt`.

Files created by this workflow:

```text
glnis.sif                         # image after a successful build
glnis.sif.lock                    # persistent output lock; leave it in place
artifacts/ubelix/build-*/build.py # frozen job helper
artifacts/ubelix/build-*/source.tar
artifacts/ubelix/build-*/build.json
artifacts/ubelix/downloads/        # Apptainer base-image download cache
logs/slurm/build/glnis-build-<job-id>.{out,err}
```

Remove old build directories only after their jobs finish. The archive and
helper remain available to diagnose or reproduce the source used by a job.
The Python base tag and unpinned Python/Git dependencies can change between
builds; the record and installed-package manifest describe what was actually
built. This is not a fully locked package environment.

For a local build on a machine with working Apptainer, run from the checkout:

```bash
apptainer build glnis.sif apptainer.def
# Optional backend:
apptainer build --build-arg WITH_GAMMALOOP=1 glnis-gammaloop.sif apptainer.def
```

Do not run those build commands on a cluster login node. Use `ubelix.py` there.
The definition uses Apptainer's normal fakeroot handling; it does not remap UIDs
itself or override the cluster's fakeroot setup.

## GPU runtime: what is needed

A working host driver is necessary, but the Python application also needs a
CUDA-enabled PyTorch build. The image supplies the latter from PyTorch's official
wheel index. The current default is PyTorch 2.14.1 with CUDA 12.6. No CUDA/cuDNN
module needs to be loaded for this image. The host driver and GPU architecture
must support the selected wheel; confirm this during the first GPU allocation.
Use `--torch-version` and `--torch-index` to select another supported combination
when building. Local builds use the corresponding `TORCH_VERSION` and
`TORCH_INDEX` build arguments. See [PyTorch's wheel combinations](https://pytorch.org/get-started/previous-versions/).

At execution, `apptainer exec --nv` provides the host's driver libraries and GPU
devices. Let Apptainer manage these bindings and its library path. Do not bind
an entire host CUDA installation, add host system library directories, or set
`TRITON_LIBCUDA_PATH` for the image. Its standard library loader and installed
C++ compiler are sufficient. This follows [Apptainer's GPU setup](https://apptainer.org/docs/user/latest/gpu.html).

With `--cleanenv`, explicitly forward Slurm's assigned device mask in the worker
job, after the allocation starts:

```bash
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?request a Slurm GPU first}"
if [ -n "${CUDA_DEVICE_ORDER:-}" ]; then
    export APPTAINERENV_CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER"
fi
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export APPTAINERENV_TORCHINDUCTOR_COMPILE_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Only when the sampler needs the optional native backend:
# export APPTAINERENV_SYMBOLICA_LICENSE="$SYMBOLICA_LICENSE"
```

This preserves the allocated devices and their ordering. Sampler `gpu_id="auto"`
chooses within that set; it does not allocate GPUs or bypass Slurm.
The thread settings keep numerical work and compiler workers within the CPU
allocation.
The application does not force 64 CPU threads at startup. The Inductor policy
sets `TRITON_DEFAULT_FP_FUSION=1` only when unset, avoiding Triton's warning about
its changing default while preserving the measured arithmetic. An explicit
`APPTAINERENV_TRITON_DEFAULT_FP_FUSION=0` is honored.

## Data and compiler-cache bindings

UBELIX network-storage paths must be bound explicitly. Resolve symlinks and bind
the actual project directory, rather than just `/scratch` or `/storage`. See
[UBELIX's filesystem-binding instructions](https://hpc-unibe-ch.github.io/runjobs/scheduled-jobs/container-jobs/#binding-network-file-systems-in-the-container).

For a GammaBoard workspace and sampler checkout that are siblings, use:

```toml
command = ["apptainer", "exec", "--cleanenv", "--nv", "--bind", "$resources/../..:$resources/../..", "$resources/../../glnis_gammaboard_api/glnis.sif", "glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

Resolve/adjust the parent path for the actual cluster layout. Bind any physics
states or checkpoint directories outside that parent separately. The
[baseline card](../../examples/templates/runs/gammaloop_train_sample.toml) uses a
virtual-environment command; replace just that command to select the image.

Compilation also needs a writable cache. Reuse it across workers of the same
software/device setup, for example:

```bash
mkdir -p /scratch/network/users/$USER/glnis-compile
CACHE=$(realpath /scratch/network/users/$USER/glnis-compile)
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/cache/inductor
export APPTAINERENV_TRITON_CACHE_DIR=/cache/triton
apptainer exec --cleanenv --nv --bind "$CACHE:/cache" glnis.sif glnis-check --device cuda:0 --compile
```

Add that same explicit cache bind to a GammaBoard process command if it uses
these environment variables. The image itself is read-only. Compiler caches are
runtime data and are not stored in sampler checkpoints. GammaBoard's current
300-second process request deadline still applies to first compilation; see
[the process guide](../../docs/usage/gammaboard.md#compilation-and-the-process-deadline).

## First UBELIX validation: still pending

No UBELIX access was available during this rewrite. Local tests check source
freezing, Slurm command construction, scratch use, failure handling and atomic
publication. They cannot establish the cluster's namespace/fakeroot policy,
compute-node network access, filesystem layout, or NVIDIA driver compatibility.

The core image passed local CPU tests and compiled GPU training/checkpoint
recovery on an NVIDIA A30 with PyTorch 2.14.1 / CUDA 12.6. Building on that NixOS
host required temporary host mount setup and an apt UID workaround; neither is
part of the cluster recipe. The unmodified recipe's rootless build on UBELIX
and the optional native image remain unverified.

When access becomes available:

1. Submit the core build first. Check the job actually runs on a compute node,
   the log's scratch path has enough node-local disk space, and the record ends
   in `completed`. Confirm Apptainer's normal fakeroot mode works with apt/pip.
2. On a CPU allocation, run `apptainer test --cleanenv glnis.sif`. For the optional
   backend, build the separate native profile, check its companion import, and
   exercise a state-backed mapping with the actual GammaLoop state and license.
3. Request one GPU, for example
   `srun --account=gratis --partition=gpu --gres=gpu:rtx3090:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash`.
   Confirm the current account/partition/type with the [GPU job guide](https://hpc-unibe-ch.github.io/runjobs/scheduled-jobs/gpus/).
4. Record `nvidia-smi`, Slurm's `CUDA_VISIBLE_DEVICES`, the PyTorch/CUDA versions,
   and `torch.cuda.get_arch_list()`. Forward the mask as above. Run
   `apptainer exec --cleanenv --nv glnis.sif glnis-check --device cuda:0`, then
   repeat with `--compile` and a persistent cache. A requested GPU check fails
   if CUDA is unavailable; it does not silently pass on CPU.
5. Repeat the compiled check in a fresh container process using the same cache.
   Check checkpoint recovery, compiler startup cost and the GammaBoard deadline.
6. Confirm data/checkpoint binds are writable where intended, and send a small
   GammaBoard training/inference task through the process protocol. Test a
   deliberately failed build and verify it leaves the previous SIF usable.

Keep the job logs and record the tested GPU, driver, image hash and package
manifest before treating an image as validated for cluster production.
