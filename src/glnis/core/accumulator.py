# type: ignore
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from time import perf_counter
from typing import List, Dict, Literal, Any
from numpy.typing import NDArray

from glnis.utils.types import LayerData
from glnis.utils.types import Result, SinglePhaseResult
from glnis.utils.helpers import Colour, error_fmter, time_fmter


class AccumulatorModule(ABC):
    """
    Calculates a number of observables for a given LayerData object. Able to combine its
    data with a AccumulatorModule of the same type.
    """

    @abstractmethod
    def combine_with(self: 'AccumulatorModule', other: 'AccumulatorModule') -> None:
        pass

    @abstractmethod
    def get_observables(self) -> Dict[str, Any]:
        return dict()

    @abstractmethod
    def str_report(self) -> str:
        return str(self.get_observables())

    def finalise(self) -> None:
        """
        Finalises the state of the AccumulatorModule, e.g. by calculating derived observables.
        Called after all combinations are done, before reporting.
        """
        pass

    @classmethod
    def cat(cls, modules: List['AccumulatorModule']) -> 'AccumulatorModule':
        if not modules:
            return modules
        if len(modules) == 1:
            return modules[0]
        module_type = type(modules[0])
        if not all(isinstance(m, module_type) for m in modules):
            raise TypeError("All modules must be of the same type to concatenate.")
        combined_module = modules[0]
        for module in modules[1:]:
            combined_module.combine_with(module)
        combined_module.finalise()
        return combined_module


class Accumulator:
    def __init__(self,
                 modules: List[AccumulatorModule],):
        self.modules = modules

    def combine_with(self: 'Accumulator', other: 'Accumulator') -> None:
        if not (type(self) is type(other) and len(self.modules) == len(other.modules)):
            raise TypeError("Cannot combine Accumulators of different types.")

        for s_module, o_module in zip(self.modules, other.modules):
            s_module.combine_with(o_module)

    def get_observables(self) -> Dict[str, Any]:
        self.finalise()
        obs = dict()
        for module in self.modules:
            obs.update(module.get_observables())
        # Histories must retain this update's values after subsequent combinations.
        return deepcopy(obs)

    def finalise(self) -> None:
        for module in self.modules:
            module.finalise()

    def str_report(self) -> str:
        self.finalise()
        line = 125*"="
        report = [line]
        for module in self.modules:
            module_report = module.str_report()
            if module_report:
                report.append(module_report)
        report.append(line)

        return "\n".join(report)

    @classmethod
    def cat(cls, accumulators: List['Accumulator']) -> 'Accumulator':
        if not accumulators:
            return accumulators
        if len(accumulators) == 1:
            return accumulators[0]
        accumulator_type = type(accumulators[0])
        if not all(isinstance(a, accumulator_type) for a in accumulators):
            raise TypeError("All accumulators must be of the same type to concatenate.")
        combined_accumulator = accumulators[0]
        for accumulator in accumulators[1:]:
            combined_accumulator.combine_with(accumulator)
        combined_accumulator.finalise()
        return combined_accumulator


class DefaultAccumulator(Accumulator):
    def __init__(self,
                 data: LayerData | None = None,
                 target: Result | None = None,
                 strat_channels: List[int] | None = None,
                 **kwargs):
        self.statistics = IntegrationStatistics(data, target=target, strat_channels=strat_channels)
        self.max_weight = MaxWeight(data)
        self.failures_monitor = FailuresMonitor(data)
        self.processing_times = ProcessingTimes(data)

        modules = [
            self.statistics,
            self.max_weight,
            self.failures_monitor,
            self.processing_times,
        ]
        super().__init__(modules)

    @property
    def result(self) -> Result:
        if self.statistics.result is None:
            self.statistics.finalise()
        return self.statistics.result


class TrainingAccumulator(DefaultAccumulator):
    def __init__(self,
                 data: LayerData,
                 training_phase: Literal['real', 'imag', 'abs'] = 'real',
                 **kwargs):
        super().__init__(data=data, **kwargs)
        total_weight = data.jac*data.wgt*data.func_val
        # Keep per-chunk arrays in a list so combine_with() can append and finalise() can concatenate.
        match training_phase:
            case 'real':
                self._training_data: List[NDArray] = [total_weight.real.ravel()]
            case 'imag':
                self._training_data: List[NDArray] = [total_weight.imag.ravel()]
            case 'abs':
                self._training_data: List[NDArray] = [np.abs(total_weight).ravel()]
            case _:
                raise ValueError(
                    "Training phase must be one of 'real', 'imag' or 'abs'.")
        self.training_result = SinglePhaseResult.from_values(
            values=self._training_data[0],
            total_time=self.processing_times.processing_times.get('total_time', 0)
        )

    @property
    def training_data(self) -> NDArray:
        if len(self._training_data) > 1:
            self.finalise()
        return self._training_data[0]

    def combine_with(self: 'TrainingAccumulator', other: 'TrainingAccumulator') -> None:
        super().combine_with(other)
        self._training_data.extend(other._training_data)
        self.training_result.combine_with(other.training_result)

    def finalise(self) -> None:
        super().finalise()
        if len(self._training_data) > 1:
            self._training_data = [np.concatenate(self._training_data)]


class IntegrationStatistics(AccumulatorModule):
    """
    Implements AccumulatorModule. Calculates the integration result.

    :var n_points:
    """

    def __init__(self,
                 data: LayerData | None = None,
                 target: Result | None = None,
                 precision: int = 2,
                 strat_channels: List[int] | None = None,):
        self.target = target or Result()
        self.precision = precision
        strat_channels = strat_channels or []
        self._result: Result = Result()

        self._finalised = False
        self._strat_dict: Dict[tuple[int], Result] = dict()
        # Build the Cartesian product of stratification channel indices.
        all_channels = [] if not strat_channels else np.array(np.meshgrid(
            *[range(dim) for dim in strat_channels]), dtype=np.uint64).T.reshape(-1, len(strat_channels))
        total_wgt = data.jac*data.wgt*data.func_val if data is not None else np.empty(0, dtype=complex)
        self._result_non_stratified = Result.from_values(
            real_values=total_wgt.real,
            imag_values=total_wgt.imag,
            total_time=data.get_processing_times().get('total_time', 0) if data is not None else 0,
        )
        total_time = data.get_processing_times().get('total_time', 0) if data is not None else 0
        for ch in all_channels:
            if data is None:
                self._strat_dict[tuple(ch)] = Result()
            else:
                ch_mask = np.all(data.strata[:, :len(strat_channels)] == ch, axis=1)
                ch_wgt = total_wgt[ch_mask]
                self._strat_dict[tuple(ch)] = Result.from_values(
                    real_values=ch_wgt.real,
                    imag_values=ch_wgt.imag,
                    total_time=total_time * ch_mask.sum() / data.n_points if data.n_points > 0 else 0,
                )

    @property
    def result(self) -> Result:
        if not self._finalised:
            self.finalise()
        if len(self._strat_dict) == 0:
            return self._result_non_stratified
        return self._result

    @property
    def result_non_stratified(self) -> Result | None:
        if not self._finalised:
            self.finalise()
        return self._result_non_stratified

    def combine_with(self: 'IntegrationStatistics', other: 'IntegrationStatistics') -> None:
        assert len(self._strat_dict) == len(other._strat_dict) and all(key in self._strat_dict for key in other._strat_dict.keys(
        )), "Stratification channels must be the same to combine IntegrationStatistics."
        self._finalised = False
        self._result_non_stratified.combine_with(other._result_non_stratified)
        for key in self._strat_dict.keys():
            s, o = self._strat_dict[key], other._strat_dict[key]
            self._strat_dict[key].combine_with(other._strat_dict[key])
            c = self._strat_dict[key]

    def str_report(self) -> str:
        if not self._finalised:
            self.finalise()
        report = []
        report.append(
            f"{Colour.PURPLE}Integration Result{Colour.END} using {Colour.BLUE}{self.result.n_points}{Colour.END} evals:")

        def add_result_str(result: Result, target: Result, comp_tgt: bool = True) -> None:
            for phase_str, res, res_abs, tgt, tgt_abs in zip(
                ('RE', 'IM'),
                (result.real, result.imag),
                (result.abs_real, result.abs_imag),
                (target.real, target.imag),
                (target.abs_real, target.abs_imag),
            ):
                if res.error <= 0:
                    continue
                report.append(f"""    {Colour.DARKCYAN}{phase_str}{Colour.END} : {
                    error_fmter(res.mean, res.error, self.precision)}""")
                err_rel = abs(res.error / res.mean) if res.mean != 0 else float('inf')
                err_col = Colour.GREEN if err_rel < 0.01 else (
                    Colour.YELLOW if err_rel < 0.05 else Colour.RED)
                report[-1] += f" ({err_col}{err_rel:.3%}{Colour.END})"
                if res.rsd > 0:
                    report[-1] += f", RSD={res.rsd:.3f}"
                if res_abs.rsd > 0:
                    report[-1] += f", ARSD={res_abs.rsd:.3f}"
                if res.tvar > 0:
                    report[-1] += f", TVAR={res.tvar:.3e}"
                if res_abs.tvar > 0:
                    report[-1] += f", ATVAR={res_abs.tvar:.3e}"

                if not comp_tgt:
                    continue
                if tgt.mean == 0:
                    continue

                diff = res.mean - tgt.mean
                sigma_diff = abs(diff) / res.error if res.error > 0 else 0
                rel_diff = abs(diff / tgt.mean)
                if tgt.error > 0:
                    target_str = error_fmter(tgt.mean, tgt.error, self.precision)
                else:
                    target_str = f"{tgt.mean:<10.6e}"

                report.append(
                    f"    tgt: {target_str} Δ = {diff:<+.{self.precision}e}")
                diff_col = Colour.GREEN if rel_diff < 0.01 else Colour.RED
                sigma_col = Colour.GREEN if abs(sigma_diff) < 1 else (
                    Colour.YELLOW if abs(sigma_diff) < 2 else Colour.RED)
                report[-1] += f" {diff_col}{rel_diff:.3%}{Colour.END}"
                report[-1] += f" {sigma_col}{sigma_diff:.2f}{Colour.END}σ" if sigma_diff > 0 else ""
                if tgt.rsd > 0:
                    report[-1] += f", RSD={tgt.rsd:.3f}"
                if tgt_abs.rsd > 0:
                    report[-1] += f", ARSD={tgt_abs.rsd:.3f}"
                if tgt.tvar > 0:
                    report[-1] += f", TVAR={tgt.tvar:.3e}"
                if tgt_abs.tvar > 0:
                    report[-1] += f", ATVAR={tgt_abs.tvar:.3e}"

        if len(self._strat_dict) > 0:
            report.append(f"{Colour.PURPLE}Stratified:{Colour.END}")
            add_result_str(self.result, self.target)
            report.append(f"{Colour.PURPLE}Non-Stratified:{Colour.END}")
            add_result_str(self.result_non_stratified, self.target)
        else:
            add_result_str(self.result, self.target)

        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        strat = len(self._strat_dict) > 0
        obs = dict(
            real_mean=self.result.real.mean,
            real_error=self.result.real.error,
            real_rsd=self.result.real.rsd,
            real_tvar=self.result.real.tvar,
            abs_real_mean=self.result.abs_real.mean,
            abs_real_error=self.result.abs_real.error,
            abs_real_rsd=self.result.abs_real.rsd,
            abs_real_tvar=self.result.abs_real.tvar,
            imag_mean=self.result.imag.mean,
            imag_error=self.result.imag.error,
            imag_rsd=self.result.imag.rsd,
            imag_tvar=self.result.imag.tvar,
            abs_imag_mean=self.result.abs_imag.mean,
            abs_imag_error=self.result.abs_imag.error,
            abs_imag_rsd=self.result.abs_imag.rsd,
            abs_imag_tvar=self.result.abs_imag.tvar,

            non_strat_real_mean=self.result_non_stratified.real.mean if strat else 0.0,
            non_strat_real_error=self.result_non_stratified.real.error if strat else 0.0,
            non_strat_real_rsd=self.result_non_stratified.real.rsd if strat else 0.0,
            non_strat_real_tvar=self.result_non_stratified.real.tvar if strat else 0.0,
            non_strat_abs_real_mean=self.result_non_stratified.abs_real.mean if strat else 0.0,
            non_strat_abs_real_error=self.result_non_stratified.abs_real.error if strat else 0.0,
            non_strat_abs_real_rsd=self.result_non_stratified.abs_real.rsd if strat else 0.0,
            non_strat_abs_real_tvar=self.result_non_stratified.abs_real.tvar if strat else 0.0,
            non_strat_imag_mean=self.result_non_stratified.imag.mean if strat else 0.0,
            non_strat_imag_error=self.result_non_stratified.imag.error if strat else 0.0,
            non_strat_imag_rsd=self.result_non_stratified.imag.rsd if strat else 0.0,
            non_strat_imag_tvar=self.result_non_stratified.imag.tvar if strat else 0.0,
            non_strat_abs_imag_mean=self.result_non_stratified.abs_imag.mean if strat else 0.0,
            non_strat_abs_imag_error=self.result_non_stratified.abs_imag.error if strat else 0.0,
            non_strat_abs_imag_rsd=self.result_non_stratified.abs_imag.rsd if strat else 0.0,
            non_strat_abs_imag_tvar=self.result_non_stratified.abs_imag.tvar if strat else 0.0
        )
        return obs

    def finalise(self) -> None:
        if self._finalised:
            return
        self._result = Result()
        for res in self._strat_dict.values():
            self._result.combine_with_stratified(res)
        self._finalised = True


class MaxWeight(AccumulatorModule):
    def __init__(self, data: LayerData | None = None,):
        if data is None or data.n_points == 0:
            self.real_pos_max_wgt: float = 0.0
            self.real_neg_max_wgt: float = 0.0
            self.imag_pos_max_wgt: float = 0.0
            self.imag_neg_max_wgt: float = 0.0
            self.real_pos_max_wgt_point = None
            self.real_neg_max_wgt_point = None
            self.imag_pos_max_wgt_point = None
            self.imag_neg_max_wgt_point = None
        else:
            total_wgt = data.jac*data.wgt*data.func_val
            cont = data.continuous
            disc = data.discrete
            real_pos_idx = total_wgt.real.argmax()
            self.real_pos_max_wgt = total_wgt.real[real_pos_idx].item()
            self.real_pos_max_wgt_point = (disc[real_pos_idx].copy(), cont[real_pos_idx].copy())
            real_neg_idx = total_wgt.real.argmin()
            self.real_neg_max_wgt = total_wgt.real[real_neg_idx].item()
            self.real_neg_max_wgt_point = (disc[real_neg_idx].copy(), cont[real_neg_idx].copy())
            imag_pos_idx = total_wgt.imag.argmax()
            self.imag_pos_max_wgt = total_wgt.imag[imag_pos_idx].item()
            self.imag_pos_max_wgt_point = (disc[imag_pos_idx].copy(), cont[imag_pos_idx].copy())
            imag_neg_idx = total_wgt.imag.argmin()
            self.imag_neg_max_wgt = total_wgt.imag[imag_neg_idx].item()
            self.imag_neg_max_wgt_point = (disc[imag_neg_idx].copy(), cont[imag_neg_idx].copy())

    def combine_with(self: 'MaxWeight', other: 'MaxWeight') -> None:
        if other.real_pos_max_wgt_point is not None:
            if self.real_pos_max_wgt_point is None or other.real_pos_max_wgt > self.real_pos_max_wgt:
                self.real_pos_max_wgt = other.real_pos_max_wgt
                self.real_pos_max_wgt_point = other.real_pos_max_wgt_point
        if other.real_neg_max_wgt_point is not None:
            if self.real_neg_max_wgt_point is None or other.real_neg_max_wgt < self.real_neg_max_wgt:
                self.real_neg_max_wgt = other.real_neg_max_wgt
                self.real_neg_max_wgt_point = other.real_neg_max_wgt_point
        if other.imag_pos_max_wgt_point is not None:
            if self.imag_pos_max_wgt_point is None or other.imag_pos_max_wgt > self.imag_pos_max_wgt:
                self.imag_pos_max_wgt = other.imag_pos_max_wgt
                self.imag_pos_max_wgt_point = other.imag_pos_max_wgt_point
        if other.imag_neg_max_wgt_point is not None:
            if self.imag_neg_max_wgt_point is None or other.imag_neg_max_wgt < self.imag_neg_max_wgt:
                self.imag_neg_max_wgt = other.imag_neg_max_wgt
                self.imag_neg_max_wgt_point = other.imag_neg_max_wgt_point

    def str_report(self) -> str:
        report = [f"{Colour.PURPLE}Max Weights{Colour.END}"]

        def mwp_to_str(mwp):
            disc, cont = mwp
            out = f"{Colour.DARKCYAN}ch{Colour.END}=[{' '.join(f"{d:.0f}" for d in disc)}] " if disc.size > 0 else ""
            out += f"{Colour.DARKCYAN}x{Colour.END}={cont}" if cont.size > 0 else ""
            return out

        if self.real_pos_max_wgt > 0.:
            report.append(
                f"    {Colour.DARKCYAN}RE[+]{Colour.END} : {self.real_pos_max_wgt:<15.12e} at {mwp_to_str(self.real_pos_max_wgt_point)}")
        if self.real_neg_max_wgt < 0.:
            report.append(
                f"    {Colour.DARKCYAN}RE[-]{Colour.END} : {self.real_neg_max_wgt:<15.12e} at {mwp_to_str(self.real_neg_max_wgt_point)}")
        if self.imag_pos_max_wgt > 0.:
            report.append(
                f"    {Colour.DARKCYAN}IM[+]{Colour.END} : {self.imag_pos_max_wgt:<15.12e} at {mwp_to_str(self.imag_pos_max_wgt_point)}")
        if self.imag_neg_max_wgt < 0.:
            report.append(
                f"    {Colour.DARKCYAN}IM[-]{Colour.END} : {self.imag_neg_max_wgt:<15.12e} at {mwp_to_str(self.imag_neg_max_wgt_point)}")
        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            real_pos_max_wgt=self.real_pos_max_wgt,
            real_neg_max_wgt=self.real_neg_max_wgt,
            imag_pos_max_wgt=self.imag_pos_max_wgt,
            imag_neg_max_wgt=self.imag_neg_max_wgt,
            real_pos_max_wgt_point=self.real_pos_max_wgt_point,
            real_neg_max_wgt_point=self.real_neg_max_wgt_point,
            imag_pos_max_wgt_point=self.imag_pos_max_wgt_point,
            imag_neg_max_wgt_point=self.imag_neg_max_wgt_point,
        )


class ProcessingTimes(AccumulatorModule):
    def __init__(self,
                 data: LayerData | None = None,
                 detailed: bool = False):
        if data is None:
            self.processing_times = dict()
            self.n_points = 0
            self._t_init = perf_counter()
        else:
            self.processing_times = data.get_processing_times()
            self.n_points = data.n_points
            self._t_init = data._t_init
        self.time_processing = 0.0
        self.time_sampler = 0.0
        self.time_integrand = 0.0
        self.time_param = 0.0
        self.time_total = 0.0
        self.detailed = detailed

    def combine_with(self: 'ProcessingTimes', other: 'ProcessingTimes') -> None:
        self.n_points += other.n_points
        for identifier, time in other.processing_times.items():
            if identifier in self.processing_times:
                self.processing_times[identifier] += time
            else:
                self.processing_times[identifier] = time

    def str_report(self) -> str:
        self.finalise()
        time_since_init = perf_counter() - self._t_init
        time_per_sample = self.time_total / self.n_points if self.n_points else 0.0
        report = [
            f"""{Colour.PURPLE}Total Time{Colour.END} : {time_fmter(time_since_init)} | {
                time_fmter(self.time_total, prefix='CPU-')} | {
                    time_fmter(time_per_sample)} / eval. Breakdown by subroutine: {Colour.END} """
        ]
        subroutine_times = []
        processing_times = dict(
            sampler=self.time_sampler,
            param=self.time_param,
            integrand=self.time_integrand,
            processing=self.time_processing
        ) if not self.detailed else self.processing_times
        for identifier, time in processing_times.items():
            if identifier == 'total_time':
                continue
            perc_time = 100. * time / self.time_total if self.time_total else 0.0
            t = time / self.n_points if self.n_points > 0 else 0.0
            if perc_time > 50.:
                subroutine_times.append(
                    f"{Colour.DARKCYAN}{identifier}{Colour.END}: {Colour.RED}{time_fmter(t)}{Colour.END} {Colour.RED}{int(perc_time)}%{Colour.END}")
            else:
                subroutine_times.append(
                    f"{Colour.DARKCYAN}{identifier}{Colour.END}: {time_fmter(t)} {int(perc_time)}%")
        report.append(" | ".join(subroutine_times))

        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        self.finalise()
        return dict(
            processing_times=self.processing_times.copy(),
            time_total=self.time_total,
            time_integrand=self.time_integrand,
            time_sampler=self.time_sampler,
            time_param=self.time_param,
            time_processing=self.time_processing,
        )

    def finalise(self) -> None:
        self.time_total = self.processing_times.get('total_time', 0)
        self.time_processing = self.processing_times.get('processing', 0)
        sampler_keys = [k for k in self.processing_times.keys() if 'sampler' in k]
        integrand_keys = [k for k in self.processing_times.keys() if 'integrand' in k]
        param_keys = [k for k in self.processing_times.keys() if 'mapping' in k]
        self.time_sampler = sum(self.processing_times.get(k, 0) for k in sampler_keys)
        self.time_integrand = sum(self.processing_times.get(k, 0) for k in integrand_keys)
        self.time_param = sum(self.processing_times.get(k, 0) for k in param_keys)


class FailuresMonitor(AccumulatorModule):
    def __init__(self,
                 data: LayerData | None = None,
                 max_display: int = 5):
        self.max_display = max_display
        if data is None:
            self.n_points = 0
            self.n_success = 0
            self.failures = dict()
        else:
            self.n_points = data.n_points
            self.n_success = data.success.sum()
            self.failures = deepcopy(data.failures)

    def combine_with(self: 'FailuresMonitor', other: 'FailuresMonitor') -> None:
        self.n_points += other.n_points
        self.n_success += other.n_success
        for identifier, failures in other.failures.items():
            if identifier in self.failures:
                self.failures[identifier] = np.vstack([self.failures[identifier], failures])
            else:
                self.failures[identifier] = failures.copy()

    def str_report(self) -> str:
        if self.n_points == self.n_success:
            return

        n_failures = self.n_points - self.n_success
        report = [
            f"{Colour.RED}WARNING: Failed {n_failures} points!{Colour.END}"]
        n_displayed = 0
        for identifier, failures in self.failures.items():
            n_curr_id = min(failures.shape[0], self.max_display-n_displayed)
            for idx in range(n_curr_id):
                report.append(f"    At '{identifier}': {failures[idx]}")
            n_displayed += n_curr_id
            if n_displayed >= self.max_display:
                break
        if n_failures > self.max_display:
            report.append(
                f"... And {Colour.RED}{n_failures-self.max_display}{Colour.END} more failed evaluations.")

        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            failures=deepcopy(self.failures),
            n_points=self.n_points,
            n_success=int(self.n_success),
            n_failures=int(self.n_points - self.n_success),
        )
