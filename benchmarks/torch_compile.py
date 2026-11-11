"""Compare eager and compiled sampler methods; no production settings are changed.

Run from the checkout with its editable environment, for example:
    .venv/bin/python benchmarks/torch_compile.py --device cuda:0 --workload sample

Each invocation compares fresh copies of the same nonuniform proposal. Use a new
TORCHINDUCTOR_CACHE_DIR for an uncached first-call measurement. CUDA timings use
wall time with synchronization, so they include Python/launch overhead.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

from gammaboard_process import RectangularDomain
from glnis.core.nn.sampler_module import MixedFlow, SamplerModule


def make_model(discrete: tuple[int, ...], continuous: int, condition: int,
               discrete_first: bool = True, mlp_layers: int = 3,
               mlp_units: int | None = None) -> MixedFlow:
    """Gaussian-example defaults; a width override applies to all MLP subnets."""
    return MixedFlow(
        RectangularDomain(discrete, continuous), dims_c=condition,
        discrete_dims_first=discrete_first,
        flow={"spline": {"bins": 10},
              "subnet": {"layers": mlp_layers, "units": 32 if mlp_units is None else mlp_units}},
        discrete={"embedding_dim": 64, "feedforward_dim": 64, "heads": 4, "layers": 1,
                  "subnet": {"layers": mlp_layers, "units": 64 if mlp_units is None else mlp_units}},
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return time.perf_counter() - start, result


def target_values(xs: np.ndarray) -> np.ndarray:
    """Cheap positive evaluator; cycle timings include both host transfers."""
    return np.exp(-np.square(xs - 0.4).sum(axis=1)) + 0.05


def prepare(model: MixedFlow, args, compiled: bool, xs, condition):
    training = args.workload in ("train", "cycle")
    model.train(training or args.context == "no-grad")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if training else None
    weights = None
    if args.workload == "train":
        with torch.no_grad():
            weights = torch.as_tensor(target_values(xs.cpu().numpy()), device=xs.device,
                                      dtype=xs.dtype) / model.prob(xs, c=condition)
    options = {"fullgraph": args.fullgraph, "dynamic": args.dynamic,
               "backend": args.backend}
    if args.backend == "inductor":
        options["mode"] = args.mode
    if compiled and args.target == "subnets":
        subnets = list(model.continuous_flow.subnets)
        if model.discrete_flow is not None:
            subnets.append(model.discrete_flow.subnet)
        for subnet in subnets:
            subnet.compile(**options)
    sample, prob = model.sample, model.prob
    if compiled and args.target == "methods":
        if args.workload in ("sample", "cycle"):
            sample = torch.compile(sample, **options)
        if args.workload in ("prob", "train", "cycle"):
            prob = torch.compile(prob, **options)

    def run():
        if args.workload in ("sample", "prob"):
            context = torch.inference_mode if args.context == "inference" else torch.no_grad
            with context():
                if args.workload == "sample":
                    return sample(args.batch_size, c=condition, return_prob=True)
                return prob(xs, c=condition)
        if args.workload == "cycle":
            if args.protect_training_samples and compiled:
                torch.compiler.cudagraph_mark_step_begin()
            # Training generation must use no_grad, not inference_mode: these
            # samples are used by autograd when feedback arrives.
            with torch.no_grad():
                generated, old_prob = sample(args.batch_size, c=condition, return_prob=True)
                if args.protect_training_samples:
                    # Outputs must survive subsequent compiled calls. A training
                    # window would retain these until its feedback is complete.
                    generated, old_prob = generated.clone(), old_prob.clone()
                host_xs = generated.cpu().numpy()
                host_prob = old_prob.cpu().numpy()
                host_weights = target_values(host_xs) / host_prob
                batch_weights = torch.as_tensor(host_weights, device=xs.device, dtype=xs.dtype)
            train_xs = generated
        else:
            train_xs, batch_weights = xs, weights
        optimizer.zero_grad(set_to_none=True)
        q = prob(train_xs, c=condition)
        loss = -(batch_weights.detach() * q.clamp_min(torch.finfo(q.dtype).tiny).log()).mean()
        loss.backward()
        optimizer.step()
        return loss

    return run, prob, sample


def check_probability_and_gradients(model, prob, xs, condition, *, backward: bool,
                                    inference: bool):
    """Compare against eager at identical parameters, including changed weights."""
    tolerance = dict(rtol=2e-4, atol=2e-5) if xs.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-9)
    model.zero_grad(set_to_none=True)
    context = nullcontext if backward else torch.inference_mode if inference else torch.no_grad
    with context():
        expected = model.prob(xs, c=condition)
        actual = prob(xs, c=condition)
        torch.testing.assert_close(actual, expected, **tolerance)
        if backward:
            named_params = tuple(model.named_parameters())
            params = tuple(parameter for _, parameter in named_params)
            expected_grad = torch.autograd.grad(expected.log().mean(), params, allow_unused=True)
            actual_grad = torch.autograd.grad(actual.log().mean(), params, allow_unused=True)
            for (name, _), actual_g, expected_g in zip(named_params, actual_grad, expected_grad):
                if expected_g is None:
                    assert actual_g is None
                else:
                    try:
                        torch.testing.assert_close(actual_g, expected_g, **tolerance)
                    except AssertionError as error:
                        difference = (actual_g - expected_g).double()
                        relative_l2 = (difference.norm() / expected_g.double().norm().clamp_min(1e-300)).item()
                        raise AssertionError(
                            f"Gradient mismatch for {name} (relative L2 error {relative_l2:.6g}): {error}"
                        ) from error


def measure(base, args, xs, condition, compiled: bool):
    if xs.is_cuda:
        torch.cuda.reset_peak_memory_stats(xs.device)
    model = deepcopy(base)
    start = time.perf_counter()
    run, prob, sample = prepare(model, args, compiled, xs, condition)
    synchronize(xs.device)
    setup = time.perf_counter() - start
    first, result = timed(run, xs.device)
    # Validate generation immediately, before another call can reuse CUDA-graph
    # storage. Eager/compiled random streams are intentionally not compared.
    if args.workload == "sample":
        with torch.no_grad():
            generated, generated_prob = result
            expected = model.prob(generated, c=condition)
            tolerance = dict(rtol=2e-4, atol=2e-5) if xs.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-9)
            torch.testing.assert_close(generated_prob, expected, **tolerance)
            assert bool(torch.isfinite(generated).all())
    del result
    for _ in range(args.warmup):
        run()
    synchronize(xs.device)
    repeats = []
    for _ in range(args.repeats):
        def repeat():
            for _ in range(args.iterations):
                run()
        elapsed, _ = timed(repeat, xs.device)
        repeats.append(elapsed / args.iterations)
    memory = ({"timed_peak_allocated_bytes": torch.cuda.max_memory_allocated(xs.device),
               "timed_peak_reserved_bytes": torch.cuda.max_memory_reserved(xs.device)}
              if xs.is_cuda else {})
    if args.workload in ("prob", "train", "cycle"):
        check_probability_and_gradients(model, prob, xs, condition,
                                        backward=args.workload in ("train", "cycle"),
                                        inference=args.context == "inference")
    result = {"setup_s": setup, "first_call_s": first, "median_s": statistics.median(repeats),
              "repeats_s": repeats, "correctness": "passed", **memory}
    if args.probe_batch_sizes:
        result["batch_shape_probes"] = []
        context = torch.inference_mode if args.context == "inference" else torch.no_grad
        for size in args.probe_batch_sizes:
            c = torch.zeros(size, args.condition, device=xs.device, dtype=xs.dtype) if args.condition else None
            with context():
                elapsed, (generated, generated_prob) = timed(
                    lambda: sample(size, c=c, return_prob=True), xs.device)
                expected = model.prob(generated, c=c)
                tolerance = dict(rtol=2e-4, atol=2e-5) if xs.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-9)
                torch.testing.assert_close(generated_prob, expected, **tolerance)
            result["batch_shape_probes"].append({"batch_size": size, "first_call_s": elapsed,
                                                  "correctness": "passed"})
    if args.check_restore:
        snapshot = model.snapshot()
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(xs.device) if xs.is_cuda else None
        with torch.no_grad():
            expected = tuple(value.clone() for value in run())
        resumed = SamplerModule.from_snapshot(snapshot).to(xs.device, xs.dtype)
        resumed_run, _, _ = prepare(resumed, args, compiled, xs, condition)
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, xs.device)
        with torch.no_grad():
            actual = resumed_run()
            for actual_value, expected_value in zip(actual, expected):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        result["module_restore"] = "passed (same execution mode and RNG state)"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--discrete", type=int, nargs="*", default=[3])
    parser.add_argument("--continuous", type=int, default=3)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--mlp-units", type=int,
                        help="hidden width of every coupling and discrete-head MLP (defaults: 32/64)")
    parser.add_argument("--recompile-limit", type=int,
                        help="override Dynamo's per-function variant limit for this benchmark process")
    parser.add_argument("--continuous-first", action="store_true")
    parser.add_argument("--protect-training-samples", action="store_true",
                        help="mark cycle boundaries and clone generated tensors outside compilation")
    parser.add_argument("--workload", choices=("sample", "prob", "train", "cycle"), default="sample")
    parser.add_argument("--context", choices=("no-grad", "eval-no-grad", "inference"), default="inference")
    parser.add_argument("--target", choices=("methods", "subnets"), default="methods")
    parser.add_argument("--mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--check-restore", action="store_true",
                        help="check exact module/RNG continuation for sampling (outside timings)")
    parser.add_argument("--probe-batch-sizes", type=int, nargs="*", default=[],
                        help="validate other sampling shapes after timing the main batch size")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.iterations, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("batch size, iterations, repeats and threads must be positive; warmup nonnegative")
    if args.mlp_layers < 1 or (args.mlp_units is not None and args.mlp_units < 1):
        parser.error("MLP layers and units must be positive")
    if args.recompile_limit is not None and args.recompile_limit < 1:
        parser.error("recompile limit must be positive")
    if args.check_restore and (args.workload != "sample" or args.target != "methods"):
        parser.error("--check-restore requires --workload sample --target methods")
    if args.probe_batch_sizes and (args.workload != "sample" or min(args.probe_batch_sizes) < 1):
        parser.error("--probe-batch-sizes requires sampling and positive sizes")
    if args.protect_training_samples and args.workload != "cycle":
        parser.error("--protect-training-samples requires --workload cycle")
    torch.set_num_threads(args.threads)
    if args.recompile_limit is not None:
        torch._dynamo.config.recompile_limit = args.recompile_limit
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.manual_seed(42)
    model = make_model(tuple(args.discrete), args.continuous, args.condition,
                       discrete_first=not args.continuous_first, mlp_layers=args.mlp_layers,
                       mlp_units=args.mlp_units).to(device, dtype)
    # A trained proposal is nonuniform; zero-initialized heads hide numerical
    # differences and do not exercise gradients through the earlier layers.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        condition = torch.randn(args.batch_size, args.condition, device=device, dtype=dtype) if args.condition else None
        xs = model.sample(args.batch_size, c=condition).detach().clone()
    result = {"torch": torch.__version__, "python": platform.python_version(),
              "device": str(device),
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "coupling_blocks": len(model.continuous_flow.subnets),
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
              "compiler_environment": {key: os.environ.get(key) for key in
                                       ("TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_COMPILE_THREADS",
                                        "TRITON_DEFAULT_FP_FUSION")},
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}}
    for label, compiled in (("eager", False), ("compiled", True)):
        print(f"Measuring {label}: {args.workload}, {args.device}, {args.dtype}, batch {args.batch_size}",
              file=sys.stderr, flush=True)
        try:
            # Give baseline/candidate the same starting RNG without advancing
            # any live sampler. Compiled RNG need not match eager RNG.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                result[label] = measure(model, args, xs, condition, compiled)
        except Exception as error:
            result[label] = {"error": type(error).__name__, "message": str(error)}
    if "median_s" in result.get("compiled", {}) and "median_s" in result.get("eager", {}):
        eager, compiled = result["eager"], result["compiled"]
        saving = eager["median_s"] - compiled["median_s"]
        result["speedup"] = eager["median_s"] / compiled["median_s"]
        startup = (compiled["setup_s"] + compiled["first_call_s"]
                   - eager["setup_s"] - eager["first_call_s"])
        result["break_even_calls"] = max(0, startup) / saving if saving > 0 else None
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)
    return 1 if any("error" in result.get(label, {}) for label in ("eager", "compiled")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
