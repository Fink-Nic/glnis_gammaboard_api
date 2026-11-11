"""Benchmark the public sampler and its native execution/checkpoint lifecycle.

Default profile: six continuous coordinates, four-layer/512-unit MLPs, float64.
Each parent run checks recovery in fresh workers with reused and empty compiler
caches. Checkpoints/reference tensors live in a temporary directory, not Git.
Bitwise continuation and f64 numerical agreement are reported separately.
"""
import time

PROCESS_START = time.perf_counter()

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import torch
from gammaboard_process import Feedback, RectangularDomain
from gammaboard_sampler import GammaBoardSampler


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start, result


def target(batch):
    # A cheap positive evaluator keeps proposal/orchestration costs visible.
    values = np.exp(-np.square(batch.xs_continuous - 0.4).sum(1)) + 0.05
    return values * (1 + 0.25 * batch.xs_discrete.sum(1))


def cycle(sampler):
    batch = sampler.generate(sampler.orchestrator.cfg.training_batch_size)
    sampler.feedback(Feedback(target(batch)))


def checkpoint_state(sampler, device):
    orchestrator = sampler.orchestrator
    return {"module": orchestrator.module.state_dict(),
            "optimizer": orchestrator.optimizer.state_dict(),
            "scheduler": orchestrator.scheduler.state_dict(),
            "training": asdict(orchestrator.training_status),
            "cpu_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "pending_mapping_weights": sampler.pending_mapping_weights}


def measure(fn, device, warmup, iterations, repeats):
    first_s, first = timed(fn, device)
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) / iterations)
    return {"first_s": first_s, "median_s": statistics.median(timings), "repeats_s": timings}, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy", choices=("eager", "default", "reduce-overhead"), default="default")
    parser.add_argument("--discrete", type=int, nargs="*", default=[3])
    parser.add_argument("--continuous", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--units", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--bulk-size", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--inference-iterations", type=int, default=100)
    parser.add_argument("--bulk-iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.resume.read_text()) if args.resume else None
    if bundle:
        args = argparse.Namespace(**{**bundle["arguments"], "resume": args.resume, "output": args.output})
    if args.batch_size < 4 or args.batch_size % 4:
        parser.error("batch-size must be a positive multiple of four")
    if min(args.iterations, args.inference_iterations, args.bulk_iterations, args.repeats,
           args.bulk_size, args.layers, args.units) < 1 or args.warmup < 0:
        parser.error("sizes/repeats must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    domain = RectangularDomain(tuple(args.discrete), args.continuous)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result = {"torch": torch.__version__, "arguments": arguments,
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR")}
    context = nullcontext(str(args.resume.parent)) if bundle else tempfile.TemporaryDirectory(prefix="glnis-native-compile-")
    with context as directory:
        root = Path(directory)
        if bundle:
            print("Restoring", args.policy, args.device, flush=True)
            result["restore_load_s"], sampler = timed(
                lambda: GammaBoardSampler.from_snapshot(snapshot=bundle["snapshot"], domain=domain), device)
        else:
            execution = {"kind": "eager"} if args.policy == "eager" else {
                "kind": "torch_compile",
                "sample": {"mode": args.policy, "dynamic": args.policy == "default"},
                "prob": {"mode": args.policy},
            }
            steps = 2 * (1 + args.warmup + args.repeats * args.iterations) + 1
            result["construct_s"], sampler = timed(lambda: GammaBoardSampler(
                domain=domain, save_path=str(root / "checkpoint.pt"),
                module={"flow": {"subnet": {"layers": args.layers, "units": args.units}, "spline": {"bins": 10}},
                        "discrete": {"embedding_dim": 64, "feedforward_dim": 64, "heads": 4, "layers": 1,
                                     "subnet": {"layers": args.layers, "units": args.units}}},
                orchestrator={"seed": 42, "training_steps": steps, "training_batch_size": args.batch_size,
                              "max_batch_size": max(args.batch_size, args.bulk_size), "use_f64": True,
                              "use_gpu": device.type == "cuda", "gpu_id": device.index or 0,
                              "execution": execution}), device)
        assert sampler.orchestrator.device == device
        result["process_start_to_ready_s"] = time.perf_counter() - PROCESS_START
        result["parameters"] = sum(p.numel() for p in sampler.orchestrator.module.parameters())
        if not bundle:
            with torch.no_grad():
                for parameter in sampler.orchestrator.module.parameters():
                    parameter.add_(0.01 * torch.randn_like(parameter))
            print("Measuring pre-checkpoint training", args.policy, args.device, flush=True)
            result["training_before"], _ = measure(lambda: cycle(sampler), device, args.warmup, args.iterations, args.repeats)
            result["prepare_partial_s"], partial = timed(lambda: sampler.generate(args.batch_size // 2), device)
            partial_values = target(partial)
            sampler.feedback(Feedback(partial_values[:args.batch_size // 4]))
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            result["snapshot_s"], snapshot = timed(sampler.snapshot, device)
            torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
            if gpu_rng is not None:
                torch.testing.assert_close(torch.cuda.get_rng_state(device), gpu_rng, rtol=0, atol=0)
            result["snapshot_bytes"] = (root / "checkpoint.pt").stat().st_size
            bundle = {"arguments": arguments, "snapshot": snapshot,
                      "pending_feedback": partial_values[args.batch_size // 4:].tolist()}
            (root / "bundle.json").write_text(json.dumps(bundle))
            torch.save(checkpoint_state(sampler, device), root / "initial.pt")
            result["snapshot_rng_unchanged"] = True
        if args.resume:
            initial = torch.load(root / "initial.pt", weights_only=False, map_location="cpu")
            torch.testing.assert_close(checkpoint_state(sampler, device), initial, rtol=0, atol=0,
                                       check_device=False, equal_nan=True)
            result["loaded_state_exact"] = True
        assert sampler.orchestrator.training_status.num_generated == args.batch_size // 2
        assert sampler.orchestrator.training_status.num_ingested == args.batch_size // 4
        result["continuation_draw_s"], continuation = timed(lambda: sampler.generate(args.batch_size // 2), device)
        start = time.perf_counter()
        sampler.feedback(Feedback(np.asarray(bundle["pending_feedback"])))
        sampler.feedback(Feedback(target(continuation)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["continuation_feedback_s"] = time.perf_counter() - start
        print("Measuring post-checkpoint training/inference", args.policy, args.device, flush=True)
        result["training_after"], _ = measure(lambda: cycle(sampler), device, args.warmup, args.iterations, args.repeats)
        assert not sampler.orchestrator.module.training
        result["inference"], inference = measure(lambda: sampler.generate(args.batch_size), device,
                                                args.warmup, args.inference_iterations, args.repeats)
        result["bulk_inference"], bulk = measure(lambda: sampler.generate(args.bulk_size), device,
                                                3, args.bulk_iterations, args.repeats)
        for batch in (inference, bulk):
            np.testing.assert_allclose(sampler.pdf(batch.xs_discrete, batch.xs_continuous) * batch.weights,
                                       np.ones(len(batch.weights)), rtol=1e-8, atol=1e-9)
        state = {"module": sampler.orchestrator.module.state_dict(),
                 "optimizer": sampler.orchestrator.optimizer.state_dict(),
                 "scheduler": sampler.orchestrator.scheduler.state_dict(),
                 "losses": sampler.orchestrator.training_status.loss_history,
                 "continuation": continuation, "inference": inference, "bulk": bulk}
        if args.resume:
            expected = torch.load(root / "expected.pt", weights_only=False, map_location="cpu")
            # Preserve both bitwise and numerical results. Fresh code generation
            # can change roundoff even with identical saved state and RNG.
            pairs = []
            for name in ("continuation", "inference", "bulk"):
                for field in ("xs_continuous", "xs_discrete", "weights"):
                    pairs.append((name + "." + field, getattr(state[name], field), getattr(expected[name], field)))
            for name, value in state["module"].items():
                pairs.append(("module", value.cpu().numpy(), expected["module"][name].numpy()))
            for index, values in state["optimizer"]["state"].items():
                for name, value in values.items():
                    pairs.append(("optimizer", value.cpu().numpy(), expected["optimizer"]["state"][index][name].numpy()))
            assert state["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
            assert state["scheduler"] == expected["scheduler"]
            pairs.append(("losses", np.asarray(state["losses"]), np.asarray(expected["losses"])))
            result["recovery_tolerance"] = {"rtol": 1e-8, "atol": 1e-9, "discrete": "exact"}
            diagnostics = result["recovery_differences"] = {}
            for group, actual, reference in pairs:
                difference = (np.not_equal(actual, reference).astype(int) if actual.dtype.kind == "b"
                              else np.abs(actual - reference))
                tolerance = 1e-9 + 1e-8 * np.abs(reference) if actual.dtype.kind == "f" else np.zeros_like(reference)
                metrics = diagnostics.setdefault(group, {"exact": True, "within_tolerance": True,
                                                         "different_elements": 0, "max_abs": 0.0})
                metrics["exact"] &= bool(np.array_equal(actual, reference))
                metrics["within_tolerance"] &= bool(np.all(difference <= tolerance))
                metrics["different_elements"] += int(np.count_nonzero(difference))
                metrics["max_abs"] = max(metrics["max_abs"], float(difference.max(initial=0)))
            result["exact_recovery"] = "passed" if all(v["exact"] for v in diagnostics.values()) else "failed"
            result["numerical_recovery"] = "passed" if all(v["within_tolerance"] for v in diagnostics.values()) else "failed"
        else:
            torch.save(state, root / "expected.pt")
            result["recovery"] = {}
            for cache_state in ("reused", "empty"):
                print("Recovering with", cache_state, "cache", args.policy, args.device, flush=True)
                output = root / (cache_state + ".json")
                env = dict(os.environ)
                if cache_state == "empty":
                    env["TORCHINDUCTOR_CACHE_DIR"] = str(root / "empty-cache")
                command = [sys.executable, str(Path(__file__).resolve()), "--resume", str(root / "bundle.json"),
                           "--output", str(output)]
                start = time.perf_counter()
                completed = subprocess.run(command, env=env, capture_output=True, text=True)
                if output.exists():
                    result["recovery"][cache_state] = json.loads(output.read_text())
                else:
                    result["recovery"][cache_state] = {"error": completed.stderr, "stdout": completed.stdout}
                if completed.returncode:
                    result["recovery"][cache_state]["error"] = completed.stderr or f"worker exited {completed.returncode}"
                result["recovery"][cache_state]["worker_wall_s"] = time.perf_counter() - start
        result["probability_check"] = "passed"
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return int(result.get("numerical_recovery") == "failed" or any(
        "error" in value or value.get("numerical_recovery") == "failed" for value in result.get("recovery", {}).values()))


if __name__ == "__main__":
    raise SystemExit(main())
