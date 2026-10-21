# type: ignore[override]
"""Local asynchronous sampling, ordered training feedback, and observable history."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from gammaboard_process import GenerationStatus, SampleBatch

from glnis.core.accumulator import DefaultAccumulator, TrainingAccumulator
from glnis.core.config import AggregatorConfig, RunConfig
from glnis.core.integrand import Integrand
from glnis.utils.types import LayerData


class SamplerAggregator:
    """Own a sampler and integrand, with bounded evaluation concurrency.

    Sampler generation and feedback run on one thread. Integrands and the
    sampler's strat_channel_transform must support concurrent evaluation of
    independent chunks. Training feedback
    contains exactly one generated draw, in its original sample order. Inference
    updates the accumulator once per completed evaluation chunk, in completion
    order. Histories contain independent snapshots, not raw training samples.

    ``accumulator`` is an optional compatible inference accumulator to extend.
    Training always starts each step with a fresh TrainingAccumulator.
    """

    def __init__(
        self,
        sampler: Any,
        integrand: Integrand,
        config: AggregatorConfig | None = None,
        *,
        accumulator: DefaultAccumulator | None = None,
    ):
        self.sampler = sampler
        self.integrand = integrand
        self.config = config or AggregatorConfig()
        get_strata = getattr(sampler, "get_strat_cardinalities", None)
        cardinalities = get_strata() if get_strata is not None else None
        if cardinalities is not None:
            if not isinstance(cardinalities, (list, tuple)) or any(
                isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size <= 0
                for size in cardinalities
            ):
                raise ValueError("sampler.get_strat_cardinalities() must return positive integer cardinalities or None")
            cardinalities = [int(size) for size in cardinalities]
        self.strat_cardinalities = cardinalities
        self.generation_batch_size = self.config.generation_batch_size
        if self.generation_batch_size is None:
            self.generation_batch_size = getattr(getattr(sampler, "cfg", None), "max_batch_size", None)
        if type(self.generation_batch_size) is not int or self.generation_batch_size <= 0:
            raise ValueError("aggregator.generation_batch_size must be positive or inherit sampler.cfg.max_batch_size")
        self.evaluation_batch_size = self.config.evaluation_batch_size or self.generation_batch_size
        if accumulator is not None:
            expected = set(product(*(range(size) for size in cardinalities))) if cardinalities else set()
            if type(accumulator) is not DefaultAccumulator or set(accumulator.statistics._strat_dict) != expected:
                raise ValueError("accumulator must be a DefaultAccumulator with the sampler's stratification channels")
        self.accumulator = accumulator if accumulator is not None else DefaultAccumulator(
            target=integrand.target, strat_channels=self.strat_cardinalities,
        )
        self.training_history: list[dict[str, Any]] = []
        self.inference_history: list[dict[str, Any]] = []
        self._running = False
        self._failed = False
        self._elapsed = 0.0
        self._batch_id = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SamplerAggregator:
        return RunConfig.from_dict(config).build()

    @classmethod
    def from_toml(cls, path: str | Path) -> SamplerAggregator:
        return RunConfig.from_toml(path).build()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any],
        config: RunConfig | dict[str, Any],
        *,
        accumulator: DefaultAccumulator | None = None,
    ) -> SamplerAggregator:
        """Restore a sampler and its integrand from checkpoint metadata and run settings.

        The sampler checkpoint retains the trained models. The run settings
        provide the integrand and aggregator configuration; histories start
        empty, and an existing inference accumulator can be supplied explicitly.
        """
        from madnis_sampler import MadnisSampler

        configuration = config if isinstance(config, RunConfig) else RunConfig.from_dict(config)
        integrand = Integrand.from_config(configuration.integrand)
        initialization = dict(configuration.sampler)
        # A comparison may save to a different location from sampler.save_path.
        # Keep subsequent checkpoints attached to the file actually restored.
        initialization["save_path"] = snapshot.get("save_path")
        sampler = MadnisSampler.from_snapshot(
            snapshot=snapshot,
            continuous_dims=integrand.continuous_dims,
            discrete_cardinalities=list(integrand.discrete_cardinalities),
            init_args=initialization,
        )
        return cls(sampler, integrand, configuration.aggregator, accumulator=accumulator)

    @property
    def pdf_continuous_dims(self) -> int:
        """Continuous dimension of the sampler's public PDF coordinates.

        Mapped samplers expect coordinates after mapping. The full sign
        strategy routes PDF queries directly to its child samplers.
        """
        sampler = getattr(self.sampler, "madnis_1", self.sampler)
        return int(getattr(sampler, "_input_continuous_dims", self.integrand.continuous_dims))

    @property
    def pdf_discrete_cardinalities(self) -> list[int]:
        """Cardinalities in PDF input order, including strata and mapping channels.

        Sign-stratified samplers take the sign index first. Mapping channels
        follow the sampler's configured conditioning order and are required
        even though generated batches omit those mapping coordinates.
        """
        child = getattr(self.sampler, "madnis_1", None)
        sampler = self.sampler if child is None else child
        cardinalities = list(getattr(
            sampler, "_all_discrete_cardinalities",
            getattr(sampler, "discrete_cardinalities", self.integrand.discrete_cardinalities),
        ))
        return cardinalities if child is None else list(self.strat_cardinalities or []) + cardinalities

    def pdf(
        self,
        xs_discrete: ArrayLike,
        xs_continuous: ArrayLike,
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate the sampler PDF in bounded batches without changing inputs.

        Inputs have shapes ``(N, len(pdf_discrete_cardinalities))`` and
        ``(N, pdf_continuous_dims)``. Densities retain the sampler's convention,
        including whether sign channels are joint or conditional densities.
        PDF queries neither generate samples nor update recorded observables.
        """
        if self._running:
            raise RuntimeError("Cannot evaluate a sampler PDF while the aggregator is running")
        evaluate = getattr(self.sampler, "pdf", None)
        if not callable(evaluate):
            raise NotImplementedError("This sampler does not provide PDF evaluation")
        maximum = getattr(getattr(self.sampler, "cfg", None), "max_batch_size", self.generation_batch_size)
        if batch_size is None:
            batch_size = maximum
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("PDF batch_size must be a positive integer")
        batch_size = min(batch_size, maximum)
        continuous = np.asarray(xs_continuous)
        if continuous.ndim != 2 or continuous.shape[1] != self.pdf_continuous_dims:
            raise ValueError(f"PDF continuous coordinates must have shape (N, {self.pdf_continuous_dims})")
        if continuous.dtype.kind not in "iuf" or not np.isfinite(continuous).all():
            raise ValueError("PDF continuous coordinates must contain finite real numbers")
        n = len(continuous)
        cardinalities = self.pdf_discrete_cardinalities
        discrete = np.asarray(xs_discrete)
        if discrete.shape != (n, len(cardinalities)):
            raise ValueError(f"PDF discrete coordinates must have shape ({n}, {len(cardinalities)})")
        if discrete.size and discrete.dtype.kind not in "iu":
            raise ValueError("PDF discrete coordinates must be integers")
        for column, cardinality in enumerate(cardinalities):
            if np.any(discrete[:, column] < 0) or np.any(discrete[:, column] >= cardinality):
                raise ValueError(f"PDF discrete column {column} must be in [0, {cardinality})")
        densities = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            # Some inverse mappings update their arrays in place.
            values = evaluate(
                discrete[start:stop].astype(np.int64, copy=True),
                continuous[start:stop].astype(np.float64, copy=True),
            )
            if values is None:
                raise NotImplementedError("This sampler does not provide PDF evaluation for these inputs")
            values = np.asarray(values)
            if values.shape != (stop - start,) or values.dtype.kind not in "iuf":
                raise ValueError("sampler.pdf() must return a real array with shape (N,)")
            densities[start:stop] = values
        return densities

    def snapshot(self, path: str | Path | None = None) -> dict[str, Any]:
        """Save the sampler and return its checkpoint metadata.

        An explicit path overrides ``sampler.cfg.save_path`` for this save
        only. Without a path, a configured save path is required. Accumulator
        results and observable histories are not part of sampler checkpoints.
        """
        if self._running:
            raise RuntimeError("Cannot checkpoint a sampler while the aggregator is running")
        save = getattr(self.sampler, "snapshot", None)
        if not callable(save):
            raise NotImplementedError("This sampler does not provide checkpoints")
        cfg = getattr(self.sampler, "cfg", None)
        previous = getattr(cfg, "save_path", None)
        destination = previous if path is None else path
        if destination is None or destination == "":
            raise ValueError("A checkpoint path or sampler.save_path is required")
        if cfg is None or not hasattr(cfg, "save_path"):
            raise NotImplementedError("This sampler does not support configurable checkpoint paths")
        cfg.save_path = str(Path(destination).expanduser().resolve())
        try:
            metadata = save()
            if not isinstance(metadata, dict) or not metadata.get("save_path"):
                raise RuntimeError("sampler.snapshot() did not create a checkpoint")
            return metadata
        finally:
            cfg.save_path = previous

    def run(self, *, mode: str | None = None, inference_samples: int | None = None) -> DefaultAccumulator:
        """Run synchronously; inside an event loop use ``await run_async()``.

        Returns the cumulative inference accumulator. Training-only results are
        available in ``training_history``. Repeated inference runs extend both
        the same accumulator and its history.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(mode=mode, inference_samples=inference_samples))
        raise RuntimeError("An event loop is already running; await aggregator.run_async() instead")

    async def run_async(
        self, *, mode: str | None = None, inference_samples: int | None = None
    ) -> DefaultAccumulator:
        mode = self.config.mode if mode is None else mode
        budget = self.config.inference_samples if inference_samples is None else inference_samples
        if mode not in {"train", "infer", "train_then_infer"}:
            raise ValueError("aggregator.mode must be train, infer, or train_then_infer")
        if type(budget) is not int or budget < 0:
            raise ValueError("aggregator.inference_samples must be a nonnegative integer")
        if self._running:
            raise RuntimeError("This aggregator is already running")
        if self._failed:
            raise RuntimeError("This aggregator failed previously; construct a new run to avoid incomplete feedback")

        self._running = True
        self._started = perf_counter()
        self._tasks: set[asyncio.Task] = set()
        self._sampler_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glnis-sampler")
        self._evaluation_pool = ThreadPoolExecutor(
            max_workers=self.config.evaluation_workers, thread_name_prefix="glnis-evaluation"
        )
        try:
            if mode in {"train", "train_then_infer"}:
                await self._train()
            if mode in {"infer", "train_then_infer"}:
                if await self._call_sampler(self.sampler.training_samples_remaining) is not None:
                    raise ValueError("Inference requires completed training (or sampler.training_steps = 0)")
                await self._infer(budget)
            self.accumulator.finalise()
            return self.accumulator
        except BaseException:
            self._failed = True
            raise
        finally:
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            # Running thread jobs cannot be cancelled. Wait for their cleanup
            # without blocking the caller's event loop.
            await asyncio.to_thread(self._evaluation_pool.shutdown, wait=True, cancel_futures=True)
            await asyncio.to_thread(self._sampler_pool.shutdown, wait=True, cancel_futures=True)
            self._tasks.clear()
            self._elapsed += perf_counter() - self._started
            self._running = False

    async def _call_sampler(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self._sampler_pool, partial(function, *args))

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        return task

    async def _generate(self, limit: int) -> tuple[SampleBatch | GenerationStatus, np.ndarray | None, float]:
        start = perf_counter()
        batch = await self._call_sampler(self.sampler.generate, limit)
        duration = perf_counter() - start
        if isinstance(batch, GenerationStatus):
            return batch, None, duration
        strata = None
        if self.strat_cardinalities is not None:
            if not isinstance(batch, tuple) or len(batch) != 2:
                raise TypeError("A stratified sampler must return (SampleBatch, strata) or GenerationStatus")
            batch, strata = batch
        if not isinstance(batch, SampleBatch):
            raise TypeError("sampler.generate() must return SampleBatch or GenerationStatus")
        n = len(batch.weights)
        if not 0 < n <= limit:
            raise ValueError("Sampler returned an empty batch or exceeded its requested sample budget")
        if np.shape(batch.weights) != (n,):
            raise ValueError("Sample weights must have shape (n_samples,)")
        if np.shape(batch.xs_continuous) != (n, self.integrand.continuous_dims):
            raise ValueError("Generated continuous coordinates do not match the integrand dimensions")
        if np.shape(batch.xs_discrete) != (n, len(self.integrand.discrete_cardinalities)):
            raise ValueError("Generated discrete coordinates do not match the integrand dimensions")
        if np.asarray(batch.weights).dtype.kind not in "iuf":
            raise ValueError("Sample weights must be real numbers")
        if np.asarray(batch.xs_continuous).dtype.kind not in "iuf":
            raise ValueError("Continuous coordinates must be real numbers")
        discrete = np.asarray(batch.xs_discrete)
        if discrete.size and discrete.dtype.kind not in "iu":
            raise ValueError("Discrete coordinates must be integers")
        if self.strat_cardinalities is not None:
            strata = np.asarray(strata)
            n_strata = len(self.strat_cardinalities)
            if strata.shape != (n, n_strata) and not (n_strata == 1 and strata.shape == (n,)):
                raise ValueError("Strata must have shape (n_samples, n_strata), or (n_samples,) for one stratum dimension")
            if strata.dtype.kind not in "iu":
                raise ValueError("Strata must contain integer channel indices")
            columns = strata.reshape(n, n_strata)
            for dimension, size in enumerate(self.strat_cardinalities):
                if np.any(columns[:, dimension] < 0) or np.any(columns[:, dimension] >= size):
                    raise ValueError(f"Stratum dimension {dimension} must contain indices in [0, {size})")
            # Keep the sampler-facing shape, including the one-dimensional sign
            # labels, and detach from any buffers reused by later generation.
            strata = strata.astype(np.int64, copy=True)
        return batch, strata, duration

    async def _train(self) -> None:
        while True:
            remaining = await self._call_sampler(self.sampler.training_samples_remaining)
            if remaining is None:
                return
            if type(remaining) is not int or remaining <= 0:
                raise RuntimeError("Sampler is waiting for training feedback that this aggregator does not own")
            # Starting mid-window would omit already-generated samples from the
            # step's observables and can make positional feedback ambiguous.
            status = getattr(self.sampler, "training_status", None)
            if status is not None and (status.num_generated or status.num_ingested):
                raise ValueError("Training must start at a step boundary")
            pending: dict[int, asyncio.Task] = {}
            step_accumulator: TrainingAccumulator | None = None
            while remaining or pending:
                while remaining and len(pending) < self.config.max_in_flight_batches:
                    batch, strata, duration = await self._generate(min(self.generation_batch_size, remaining))
                    if batch is GenerationStatus.WAITING:
                        if not pending:
                            raise RuntimeError("Sampler returned WAITING with no outstanding evaluations")
                        break
                    if batch is GenerationStatus.FINISHED:
                        raise RuntimeError("Sampler finished during an incomplete training step")
                    # if batch.training_remaining != remaining:
                    #     raise RuntimeError("Sampler training window changed before ordered feedback completed")
                    remaining -= len(batch.weights)
                    batch_id = self._batch_id
                    self._batch_id += 1
                    pending[batch_id] = self._spawn(self._evaluate_training_draw(batch, duration, strata))

                # Dict insertion order is generation order, regardless of which
                # evaluation task finishes first. Each task restores chunk order.
                batch_id = next(iter(pending))
                task = pending.pop(batch_id)
                accumulator = await task
                self._tasks.discard(task)
                await self._call_sampler(self.sampler.feedback, accumulator.training_data)
                if step_accumulator is None:
                    step_accumulator = accumulator
                else:
                    step_accumulator.combine_with(accumulator)

            next_remaining = await self._call_sampler(self.sampler.training_samples_remaining)
            if next_remaining == 0:
                raise RuntimeError("Sampler did not complete its training step after receiving all feedback")
            step_accumulator.finalise()
            status = getattr(self.sampler, "training_status", None)
            record = self._record(step_accumulator, "training", len(self.training_history) + 1)
            record["step"] = getattr(status, "step", len(self.training_history) + 1)
            record["loss"] = getattr(status, "last_loss", None)
            record["training_phase"] = self.config.training_phase
            record["training_result"] = {
                name: float(getattr(step_accumulator.training_result, name))
                for name in ("mean", "error", "rsd", "tvar")
            }
            self.training_history.append(record)
            if self.config.log_every and record["step"] % self.config.log_every == 0:
                print(f"Training step {record['step']} | loss={record['loss']}", flush=True)
                print(step_accumulator.str_report(), flush=True)

    async def _evaluate_training_draw(
        self, batch: SampleBatch, duration: float, strata: np.ndarray | None = None,
    ) -> TrainingAccumulator:
        tasks = [self._spawn(self._evaluate_chunk(batch, start, stop, duration, training=True, strata=strata))
                 for start, stop in self._slices(len(batch.weights))]
        # gather preserves the original slice offsets even if workers finish in
        # reverse order. cat concatenates the feedback vectors in that order.
        chunks = await asyncio.gather(*tasks)
        self._tasks.difference_update(tasks)
        return TrainingAccumulator.cat(chunks) 

    async def _infer(self, remaining: int) -> None:
        pending: dict[asyncio.Task, tuple[int, int, int]] = {}
        draws: dict[int, int] = {}
        while remaining or pending:
            while remaining and len(draws) < self.config.max_in_flight_batches:
                batch, strata, duration = await self._generate(min(self.generation_batch_size, remaining))
                if isinstance(batch, GenerationStatus):
                    raise RuntimeError(f"Sampler returned {batch.value} before the inference budget was exhausted")
                if batch.training_remaining is not None:
                    raise RuntimeError("Sampler requested training feedback during inference")
                remaining -= len(batch.weights)
                batch_id = self._batch_id
                self._batch_id += 1
                slices = list(self._slices(len(batch.weights)))
                draws[batch_id] = len(slices)
                for start, stop in slices:
                    task = self._spawn(self._evaluate_chunk(batch, start, stop, duration, training=False, strata=strata))
                    pending[task] = batch_id, start, stop

            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                batch_id, start, stop = pending.pop(task)
                chunk = task.result()
                self._tasks.discard(task)
                self.accumulator.combine_with(chunk)
                self.accumulator.finalise()
                record = self._record(self.accumulator, "inference", len(self.inference_history) + 1)
                record.update(batch_id=batch_id, sample_start=start, sample_stop=stop, update_samples=stop - start)
                self.inference_history.append(record)
                draws[batch_id] -= 1
                if not draws[batch_id]:
                    del draws[batch_id]

    def _slices(self, n: int):
        for start in range(0, n, self.evaluation_batch_size):
            yield start, min(start + self.evaluation_batch_size, n)

    async def _evaluate_chunk(self, batch, start, stop, duration, *, training, strata=None):
        return await asyncio.get_running_loop().run_in_executor(
            self._evaluation_pool,
            partial(self._evaluate, batch, start, stop, duration, training=training, strata=strata),
        )

    def _evaluate(self, batch, start, stop, duration, *, training, strata=None):
        n = stop - start
        data = self._init_layer_data(n)
        data.continuous = batch.xs_continuous[start:stop]
        data.discrete = batch.xs_discrete[start:stop]
        data.wgt = batch.weights[start:stop]
        chunk_strata = None if strata is None else strata[start:stop].copy()
        if chunk_strata is not None:
            data.strata = chunk_strata
        data.update("processing")
        # Mapping is owned by the sampler. Its Jacobian is already in weights.
        data._processing_times["sampler"] = duration * n / len(batch.weights)
        valid = data.success.copy()
        values = np.zeros(n, dtype=np.complex128)
        if valid.any():
            evaluated = np.asarray(self.integrand.evaluate_batch(data.continuous[valid], data.discrete[valid]))
            if evaluated.shape != (int(valid.sum()),):
                raise ValueError("Integrand must return one value per sample with shape (n_samples,)")
            values[valid] = evaluated
        if chunk_strata is not None:
            # Select channels from unweighted values using the configured phase
            # in both training and inference. Rejection removes both components
            # of the complex contribution, not just the selected component.
            with np.errstate(invalid="ignore", over="ignore"):
                phase_values = np.abs(values) if self.config.training_phase == "abs" else getattr(values, self.config.training_phase).copy()
                selected = np.asarray(self.sampler.strat_channel_transform(phase_values, chunk_strata))
            if selected.shape != (n,) or selected.dtype.kind not in "iuf":
                raise ValueError("strat_channel_transform must return one real value per sample with shape (n_samples,)")
            values[selected == 0] = 0
        # Keep nonfinite evaluations (including weight-product overflow) in the
        # sample count as flagged zero contributions, as LayerData specifies.
        with np.errstate(invalid="ignore", over="ignore"):
            weighted = values * data.wgt.ravel()
            valid_product = np.isfinite(weighted)
            if training and self.config.training_phase == "abs":
                valid_product &= np.isfinite(np.abs(weighted))
        values[~valid_product] = np.nan
        data.func_val = values
        data.update("integrand")
        if training:
            return TrainingAccumulator(data, target=self.integrand.target, training_phase=self.config.training_phase,
                                       strat_channels=self.strat_cardinalities)
        return DefaultAccumulator(data, target=self.integrand.target, strat_channels=self.strat_cardinalities)

    def _init_layer_data(self, n: int) -> LayerData:
        return LayerData(
            n_points=n,
            n_cont=self.integrand.continuous_dims,
            n_disc=len(self.integrand.discrete_cardinalities),
            n_strata=len(self.strat_cardinalities or []),
        )

    def _record(self, accumulator, phase: str, update: int) -> dict[str, Any]:
        return dict(
            phase=phase,
            update=update,
            n_samples=accumulator.result.n_points,
            elapsed_time=self._elapsed + perf_counter() - self._started,
            observables=accumulator.get_observables(),
        )
