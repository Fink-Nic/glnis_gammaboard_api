"""Exercise real helper/job scripts with fake Slurm and Apptainer commands."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class UbelixBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.checkout = self.base / "checkout with spaces"
        (self.checkout / "ops/ubelix").mkdir(parents=True)
        shutil.copy2(ROOT / "ubelix.py", self.checkout / "ubelix.py")
        shutil.copy2(ROOT / "ops/ubelix/build.sbatch", self.checkout / "ops/ubelix/build.sbatch")
        for name in ("apptainer.def", "flake.nix", "flake.lock", "pyproject.toml",
                     "Cargo.toml", "Cargo.lock"):
            (self.checkout / name).write_text(f"fixture {name}\n")
        (self.checkout / "src").mkdir()
        (self.checkout / "src/sampler.py").write_text("queued source\n")
        (self.checkout / "src/dev.so").write_bytes(b"do not package a local extension")

        self.bin_dir = self.base / "bin"
        self.bin_dir.mkdir()
        self.record = self.base / "submission.json"
        self.calls = self.base / "apptainer.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin_dir}:{os.environ['PATH']}",
                        GLNIS_TEST_SUBMISSION=str(self.record), GLNIS_TEST_CALLS=str(self.calls))
        self.executable("sbatch", '''
import json, os, sys
from pathlib import Path
if os.environ.get("GLNIS_TEST_SUBMIT_FAIL"):
    sys.exit("submission rejected")
Path(os.environ["GLNIS_TEST_SUBMISSION"]).write_text(json.dumps(sys.argv[1:]))
print("12345;ubelix")
''')
        self.executable("apptainer", '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["GLNIS_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps({"args": args, "cwd": str(Path.cwd()),
                            "tmp": os.environ["APPTAINER_TMPDIR"]}) + "\\n")
if args == ["--version"]:
    print("apptainer version test")
elif args[0] == "build":
    Path(args[-2]).write_bytes(Path("src/sampler.py").read_bytes())
    if os.environ.get("GLNIS_TEST_FAIL") == "build":
        sys.exit("build failed")
elif args[0] == "test":
    assert args[1] == "--cleanenv"
    assert Path(args[-1]).is_file()
    if os.environ.get("GLNIS_TEST_FAIL") == "test":
        sys.exit("smoke test failed")
else:
    sys.exit("unexpected Apptainer command")
''')

    def executable(self, name, source):
        path = self.bin_dir / name
        path.write_text(f"#!{sys.executable}\n{source}")
        path.chmod(0o755)

    def submit(self, *options, check=True):
        return subprocess.run(
            [sys.executable, str(self.checkout / "ubelix.py"),
             "--scratch", str(self.base / "scratch with spaces"), *options],
            cwd=self.base, env=self.env, text=True, capture_output=True, check=check,
        )

    def job_arguments(self):
        arguments = json.loads(self.record.read_text())
        script_index = next(i for i, arg in enumerate(arguments) if arg.endswith("build.sbatch"))
        return arguments[:script_index], arguments[script_index:]

    def run_job(self, check=False):
        options, arguments = self.job_arguments()
        cpus = next(arg.split("=", 1)[1] for arg in options if arg.startswith("--cpus-per-task="))
        return subprocess.run(
            ["bash", *arguments], cwd=self.base,
            env=dict(self.env, SLURM_JOB_ID="12345", SLURM_CPUS_PER_TASK=cpus),
            text=True, capture_output=True, check=check,
        )

    def test_queued_build_uses_snapshot_and_publishes_tested_image(self):
        result = self.submit("--cpus", "8", "--mem", "64G", "--time", "12:00:00",
                             "--account", "my-account", "--partition", "my-partition", "--qos", "my-qos")
        self.assertIn("Submitted build job 12345", result.stdout)
        options, arguments = self.job_arguments()
        for option in ("--cpus-per-task=8", "--mem=64G", "--time=12:00:00",
                       "--account=my-account", "--partition=my-partition", "--qos=my-qos"):
            self.assertIn(option, options)
        self.assertTrue((self.checkout / "logs/slurm/build").is_dir())
        snapshot = Path(arguments[1])
        self.assertFalse((snapshot / "src/dev.so").exists())
        (self.checkout / "src/sampler.py").write_text("edited after submission\n")
        self.run_job(check=True)

        image = self.checkout / "glnis.sif"
        self.assertEqual(image.read_text(), "queued source\n")
        metadata = json.loads((self.checkout / "glnis.sif.meta.json").read_text())
        self.assertEqual(metadata["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
        self.assertEqual(metadata["source_snapshot"], str(snapshot))
        self.assertEqual(metadata["slurm_job_id"], "12345")
        self.assertEqual(metadata["build_cores"], 8)
        self.assertEqual(metadata["build_profile"], "release")
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        build = next(call for call in calls if call["args"][0] == "build")
        self.assertIn("BUILD_CORES=8", build["args"])
        self.assertEqual(build["cwd"], str(snapshot))
        self.assertTrue(any(call["args"][0] == "test" for call in calls))
        self.assertEqual(list((self.base / "scratch with spaces/tmp").iterdir()), [])

    def test_failures_preserve_previous_image_and_metadata(self):
        self.submit()
        image = self.checkout / "glnis.sif"
        metadata = self.checkout / "glnis.sif.meta.json"
        image.write_text("previous working image")
        metadata.write_text("previous build metadata")
        for phase in ("build", "test"):
            with self.subTest(phase=phase):
                self.env["GLNIS_TEST_FAIL"] = phase
                result = self.run_job()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(image.read_text(), "previous working image")
                self.assertEqual(metadata.read_text(), "previous build metadata")
                self.assertEqual(list(self.checkout.glob("*.sif.build-*")), [])
                self.assertEqual(list((self.base / "scratch with spaces/tmp").iterdir()), [])

    def test_competing_build_fails_without_replacing_image(self):
        self.submit()
        image = self.checkout / "glnis.sif"
        image.write_text("previous working image")
        with (self.checkout / "glnis.sif.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_job()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another build is writing", result.stderr)
        self.assertEqual(image.read_text(), "previous working image")
        self.assertFalse(self.calls.exists())

    def test_custom_image_path(self):
        self.submit("--output", "images/custom sampler.sif")
        self.run_job(check=True)
        self.assertTrue((self.checkout / "images/custom sampler.sif").is_file())
        self.assertTrue((self.checkout / "images/custom sampler.sif.meta.json").is_file())
        self.assertFalse((self.checkout / "glnis.sif").exists())

    def test_dry_run_does_not_create_files_or_submit(self):
        result = self.submit("--dry-run")
        self.assertIn("--cpus-per-task=4", result.stdout)
        self.assertFalse(self.record.exists())
        self.assertFalse((self.checkout / "artifacts").exists())
        self.assertFalse((self.checkout / "logs").exists())

    def test_rejected_submission_removes_unused_snapshot(self):
        self.env["GLNIS_TEST_SUBMIT_FAIL"] = "1"
        result = self.submit(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("submission rejected", result.stderr)
        self.assertEqual(list((self.checkout / "artifacts/ubelix").iterdir()), [])

    def test_invalid_cpu_count_never_submits(self):
        result = self.submit("--cpus", "0", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integer", result.stderr)
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
