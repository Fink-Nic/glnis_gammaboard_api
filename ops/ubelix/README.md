# Build the sampler on UBELIX

This workflow builds only the glNIS GammaBoard sampler image. GammaBoard and
GammaLoop use their own installation workflows; their builds can run independently.

## Clone and submit

On a UBELIX login node, clone this repository next to the GammaBoard workspace.
Choose a shared storage directory visible to the login and compute nodes:

```bash
cd /storage/research/itp_localunitaritydata/<your-parent-directory>
git clone https://github.com/Fink-Nic/glnis_gammaboard_api.git
cd glnis_gammaboard_api
# Optional: select a branch, tag, or exact commit before submitting.
# git checkout <branch-tag-or-commit>
python3 ubelix.py
```

Use a revision containing `ubelix.py`. Python 3.9 or newer and Slurm's `sbatch`
are needed on the login node. The job also needs `apptainer` and `flock` on its
PATH; use the same Apptainer environment as for the GammaBoard builds. The
Python interpreter used to submit must also be accessible on the compute node.
No sampler Python packages, host Nix, or host Rust installation are required.

The helper uses only Python's standard library. It snapshots the current
checkout's build inputs, including local edits, and immediately submits a Slurm
job. It does not fetch another repository checkout. Edits and `git checkout`
after submission do not affect the queued build. Dependency sources are fetched
according to `flake.lock` and `Cargo.lock` during the image build; compute nodes
therefore need network access, as with the other image builds.

By default, the job requests one node, one task, four CPUs, 32 GB memory, and
six hours, using account `gratis` and partition `epyc2`. Override these values
when submitting, for example:

```bash
python3 ubelix.py --cpus 8 --mem 64G --time 12:00:00
python3 ubelix.py --account <account> --partition <partition> --qos <qos>
python3 ubelix.py --dry-run
```

`--dry-run` prints the submission command without copying files or submitting a
job. `python3 ubelix.py --help` lists all options. The helper locates the checkout
from its own path, so it can also be invoked from another working directory.

## Monitor the build

The helper prints the job ID and log paths. Substitute the returned job ID:

```bash
squeue -j <job-id>
tail -f logs/slurm/build/glnis-build-<job-id>.out
tail -f logs/slurm/build/glnis-build-<job-id>.err
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
sstat -j <job-id>.batch --format=JobID,AveCPU,MaxRSS,AveRSS
```

The job builds the existing `apptainer.def`, using the allocated CPU count for
`BUILD_CORES`. Nix runs inside the build container and compiles the Rust
extension in release mode. It preserves the repository's pinned Python and
CUDA dependencies. No GPU allocation is needed for the build or its import test.
The sampler keeps Nix's native libraries ahead of automatically discovered GPU
driver directories. General system library directories are probed for drivers
without being added to `LD_LIBRARY_PATH`.

CUDA dependencies such as the Python bindings, NCCL, and NVSHMEM can compile from
source. These steps can take tens of minutes each, with only a `building ...drv`
line in the log until the step finishes. Use `sstat` to check activity and memory
use during these quiet phases. A fresh build may take several hours.
Nix build logs are streamed to the Slurm error log. Completed local builds are
saved automatically in `<scratch>/nix-cache` and reused by later submissions,
including retries after a failed build. This private file cache is trusted only
inside the build container; no host Nix configuration is changed.

The job uses `--ignore-fakeroot-command` to avoid injecting UBELIX's host
`fakeroot` executable into the Nix image, where its loader/libraries are not
available. Apptainer still uses a root-mapped user namespace. The definition sets
Nix's `build-users-group` to an empty value so it does not attempt to switch to
unmapped `nixbld` user IDs. No administrator changes to UID mappings are needed.
For the runtime build it remaps that single UID to UID 1000 in a nested user
namespace. Archive tools then behave as for an ordinary user, avoiding attempts
to restore unmapped owners from CUDA source archives. The `unshare` tool comes
from the pinned Nix package set, not a host installation.
The Rust helpers, Rust builds, and project Python build use Nixpkgs'
`writableTmpDirAsHomeHook` to keep home-directory caches in their build directory.
This prevents creation of `/homeless-shelter`, whose existence makes Nix reject
unsandboxed builds.
It also adds GPFS's `system.gpfs_nfs4_acl` to Nix's ignored ACLs, keeping Nix
from trying to remove this filesystem-managed attribute when its store is on
UBELIX shared storage. This extends Nix's existing ignored ACL list.

After building, the job runs the definition's `%test` with a clean environment
to check the sampler imports and Rust extension. Only a successful image replaces
`glnis.sif`. A failed build or smoke test leaves an existing image in place.
The smoke test uses a temporary home and cache inside the build staging directory,
so imports do not write caches or Symbolica permit locks into your home directory.
Concurrent jobs targeting the same output cannot build over one another: a
second job exits with an explanatory log message.

Default files, relative to the checkout:

```text
glnis.sif                          # sampler runtime
glnis.sif.meta.json                # commit, source hash, resources, image hash, timing
glnis.sif.lock                     # persistent build lock; no action needed
artifacts/ubelix/build-*/source/   # retained build-input snapshots
artifacts/ubelix/build-*/submission.json
logs/slurm/build/glnis-build-<job-id>.{out,err}
```

Use `--output <path>` to select another image and `--scratch <directory>` to
change Apptainer's scratch/cache location. Relative paths resolve against the
checkout. The default scratch root is
`/scratch/network/users/$USER/glnis-apptainer`, with a shared download cache and
Nix build cache, plus separate temporary directories cleaned up after each job.
Retained source snapshots can be removed after their jobs finish; do not remove
queued or running jobs' snapshots.

## Use from GammaBoard

With the GammaBoard workspace and `glnis_gammaboard_api` as siblings, the existing
Apptainer command examples point to the default output:

```toml
command = ["apptainer", "exec", "--no-mount", "/etc/localtime", "--nv", "--bind", "$resources/../..:$resources/../..", "$resources/../../glnis_gammaboard_api/glnis.sif", "glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

Run cards such as `examples/templates/runs/ghost_bump_madnis.toml` and
`examples/templates/runs/gg_hhh_1L.toml` include this command as a commented
alternative to their default Nix command. Select the Apptainer command in the
run card before submitting it. CUDA execution requires a GPU worker allocation
and `--nv`; it does not require rebuilding the image on a GPU node. Supply your
Symbolica license in the environment used to launch the sampler, as described
in the repository README.
