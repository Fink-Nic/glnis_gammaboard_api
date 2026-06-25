# type: ignore
import numpy as np
import functools
from time import perf_counter
from numpy.typing import NDArray, DTypeLike
from dataclasses import dataclass, field
from typing import Dict, Iterable, List
from glnis.utils.helpers import chunks


@dataclass
class ParameterisationLayerConfig:
    """
    Config for one layer of the parameterisation stack. This is used to construct the
    'LayeredParameterisation' object used to transform the sampled points before
    passing to the evaluator.
    """
    param_type: str
    param_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ParameterisationLayerConfig:
        param_type = config_dict.get("param_type", None)
        if param_type is None:
            raise ValueError("Parameterisation config must have a 'param_type' key.")
        return cls(
            param_type=config_dict.pop("param_type"),
            param_kwargs=config_dict or {},
        )


type LayerResult = NDArray | List | None


@dataclass
class GraphProperties:
    """
    Holds all essential properties of a Feynman graph, including edge information, graph signature, and orientation details.
    Attributes:
        edge_src_dst_vertices (List[List[int]]): List of source and destination vertices for each edge.
        edge_masses (List[float]): List of masses for each edge.
        edge_momentum_shifts (List[List[float]]): List of momentum shifts for each edge.
        graph_external_vertices (List[int]): List of external vertices in the graph.
        graph_signature (List[List[int]]): Signature of the graph, representing its structure.
        lmb_array (NDArray): Array representing loop momentum basis transformations.
        edge_external_sigs (List[List[float]]): External signatures associated with edges.
        external_momenta (List[List[float]]): External momenta for the graph.
        orientation_ids (List[int]): Identifiers for different orientations of the graph.
        orientation_signatures (List[List[int]]): Signatures corresponding to each orientation.
        generation_channel_id (int): Identifier for the generation channel of the graph.
        e_cm (float): Center-of-mass energy for the graph.
        n_loops (int): Number of loops in the graph, derived from the graph signature.
        n_edges (int): Number of edges in the graph, derived from the edge masses.
        edge_ismassive (List[bool]): List indicating whether each edge is massive
        n_channels (int): Number of channels, derived from the shape of the lmb_array.
        n_orientations (int): Number of orientations, derived from the length of orientation_ids
        channel_transforms (NDArray): Array of loop momentum basis transformations for each channel.
        channel_inv_transforms (NDArray): Array of inverse loop momentum basis transformations for each channel
    """
    edge_src_dst_vertices: List[List[int]]
    edge_masses: List[float]
    edge_momentum_shifts: List[List[float]]
    graph_external_vertices: List[int]
    graph_signature: List[List[int]]
    lmb_array: NDArray = field(default_factory=list)
    edge_external_sigs: List[List[float]] = field(default_factory=list)
    external_momenta: List[List[float]] = field(default_factory=list)
    orientation_ids: List[int] = field(default_factory=list)
    orientation_signatures: List[List[int]] = field(default_factory=list)
    generation_channel_id: int = 0
    e_cm: float = 0.0

    def __post_init__(self: 'GraphProperties'):
        if len(self.orientation_ids) != len(self.orientation_signatures):
            raise ValueError("Length of orientation_ids and orientation_signatures must match.")
        TOLERANCE = 1E-10
        self.n_loops: int = len(self.graph_signature[0])
        self.n_edges: int = len(self.edge_masses)
        self.edge_ismassive: list[bool] = [
            mass > TOLERANCE for mass in self.edge_masses]
        self.lmb_array = np.array(self.lmb_array, dtype=np.uint64)
        self.n_channels = self.lmb_array.shape[0]
        self.n_orientations = len(self.orientation_ids)

        try:
            # Calculate the inverse lmb transforms, ordered as the LMBs in graph properties
            self.channel_transforms = np.array(
                self.graph_signature)[self.lmb_array].reshape(self.n_channels, self.n_loops, self.n_loops)
            # Inverse transforms of each channel
            self.channel_inv_transforms = np.linalg.inv(self.channel_transforms)
        except:
            self.channel_transforms = np.zeros((0, self.n_loops, self.n_loops), dtype=np.float64)
            self.channel_inv_transforms = np.zeros((0, self.n_loops, self.n_loops), dtype=np.float64)


class LayerData:
    """
    Holds the data to be passed along the evaluation chain. Data is stored as a contiguous numpy array, and
    accessed via property getters and setters. Getters return a view, except for `discrete`, which is cast to uint64.
    The state of the object is only updated when update() is called, which also keeps track of the time since the last update,
    and deals with non-finite values, and stores a copy of the state that caused those values.
    Getting a property that has been set, but not updated, will trigger an update.
    Attributes:
        jac (NDArray[float], shape=(n, 1)): Jacobian of the transformation from x-space to the
            integration variables for each sample.
        wgt (NDArray[float], shape=(n, 1)): Weight of each sample (e.g. from importance sampling).
        f_real (NDArray[float], shape=(n, 1)): Real part of the function value at each sample.
        f_imag (NDArray[float], shape=(n, 1)): Imaginary part of the function value at each sample.
        momenta (NDArray[float], shape=(n, m)): Momenta at which the function is evaluated; `m`
            denotes the number of momentum components tracked per sample.
        continuous (NDArray[float], shape=(n, c)): Continuous coordinates of the samples; `c` denotes
            the number of continuous features per sample.
        discrete (NDArray[int], shape=(n, d)): Discrete coordinates of the samples; `d` denotes the
            number of discrete features per sample. Accessors return this as unsigned integers
            (cast to `uint64`).
    """
    POSITIONS: Dict[str, int] = dict(
        jac=0,
        wgt=1,
        f_real=2,
        f_imag=3,
        momenta=4,
        continuous=5,
        discrete=6,
    )

    def __init__(
            self: 'LayerData',
            n_points: int = 0,
            n_mom: int = 0,
            n_cont: int = 0,
            n_disc: int = 0,
            _existing_data: NDArray | None = None,
            _existing_structure: NDArray | None = None,
            _existing_active_structure: NDArray | None = None,
            dtype: DTypeLike | None = None,
    ):
        self._timestamp = perf_counter()
        self._t_init = perf_counter()

        self._pending_data: Dict[str, NDArray] = dict()
        self._structure = np.zeros(len(LayerData.POSITIONS), dtype=np.uint32)

        # Initializing from existing LayerData object data
        if not (_existing_data is None or _existing_structure is None):
            self._data = _existing_data.copy()
            self.dtype = self._data.dtype
            self._structure = _existing_structure.copy()
            if _existing_active_structure is None:
                self._active_structure = self._structure.copy()
            else:
                self._active_structure = _existing_active_structure.copy()
            self.n_points = self._data.shape[0]
            self.success = np.isfinite(self._data).all(axis=1)
        else:
            self.n_points = int(n_points)
            self.dtype = np.dtype(np.float64) if dtype is None else dtype
            # Set the dimensions of the data
            self._structure[LayerData.POSITIONS['jac']] = 1
            self._structure[LayerData.POSITIONS['wgt']] = 1
            self._structure[LayerData.POSITIONS['f_real']] = 1
            self._structure[LayerData.POSITIONS['f_imag']] = 1
            self._active_structure = self._structure.copy()
            self._structure[LayerData.POSITIONS['momenta']] = n_mom
            self._structure[LayerData.POSITIONS['continuous']] = n_cont
            self._structure[LayerData.POSITIONS['discrete']] = n_disc
            self._data = np.zeros(
                (self.n_points, self._structure.sum()), dtype=self.dtype)
            self._data[:, LayerData.POSITIONS['jac']] = 1
            self._data[:, LayerData.POSITIONS['wgt']] = 1
            self.success = np.ones(self.n_points, dtype=np.bool)

        self.failures: Dict[str, NDArray] = dict()
        self._processing_times = dict(
            processing=perf_counter() - self._timestamp)
        self._timestamp = perf_counter()

    @staticmethod
    def _timer(func):
        @functools.wraps(func)
        def wrapper_timer(self: 'LayerData', *args, **kwargs):
            start_time = perf_counter()
            value = func(self, *args, **kwargs)
            self._processing_times['processing'] += perf_counter() - start_time
            return value

        return wrapper_timer

    @_timer
    def update(self, identifier: str = 'unspecified') -> None:
        """
        Updates state according to data in _pending_data. Will update the processing time of the
        processing step 'identifier' based on the elapsed time since last update. Will also mask
        non-finite values and update failures accordingly.
        """
        if len(self._pending_data) == 0:
            self._update_processing_times(identifier)
            return

        next_success = np.ones(self.n_points, dtype=np.bool)
        for v in self._pending_data.values():
            next_success = np.logical_and(
                next_success, np.isfinite(v).all(axis=1))

        new_failures_mask = np.logical_and(
            self.success, ~next_success)
        self.success = next_success

        if new_failures_mask.any():
            caused_failures = self._data[new_failures_mask]
            if identifier in self.failures:
                self.failures[identifier+"_"] = caused_failures
            else:
                self.failures[identifier] = caused_failures

        # Converting from iter to list allows us to remove items during loop
        for name in list(self._pending_data.keys()):
            idx = LayerData.POSITIONS[name]
            offset = self._structure[:idx].sum()
            value = self._pending_data.pop(name)
            dim = value.shape[1]
            self._active_structure[idx] = dim
            self._data[:, offset:offset+dim] = value

        self._data[~self.success] = 0

        # Update processing times according to time since last update
        self._update_processing_times(identifier)

    def get_processing_times(self) -> Dict[str, float]:
        if len(self._pending_data) > 0:
            self.update('processing_times_getter')
        processing_times = dict(self._processing_times)
        processing_times['total_time'] = sum(t for t in processing_times.values())
        return processing_times

    def wake(self) -> None:
        """
        Updates the timestamp to the current time without any other side-effects.
        Useful when the LayerData object sits idle, e.g. after retrieving it from a Queue.
        """
        self._timestamp = perf_counter()

    @_timer
    def as_chunks(self, n_chunks: int, n_cores: int = 1) -> Iterable['LayerData']:
        """
        Yields the data in n_chunks chunks, as LayerData objects. If the data is split across multiple cores,
        but was sampled on a single core, use the n_cores argument to properly keep track of relative CPU time.
        """
        if len(self._pending_data) > 0:
            self.update('as_chunks')

        data_chunks: Iterable[NDArray] = chunks(self._data, n_chunks)
        for chunk_id, data_chunk in enumerate(data_chunks):
            chunk = LayerData(
                _existing_data=data_chunk,
                _existing_structure=self._structure,
                _existing_active_structure=self._active_structure,
            )
            chunk._t_init = self._t_init
            # We add the metadata to the first chunk, and duplicate the processing time n_cores times
            if chunk_id == 0:
                chunk.failures = self.failures
            if chunk_id < n_cores:
                chunk._processing_times = self._processing_times

            yield chunk

    @property
    def jac(self: 'LayerData'):
        return self._get_data('jac')

    @jac.setter
    def jac(self: 'LayerData', value: LayerResult):
        self._set_data('jac', value)

    @property
    def wgt(self: 'LayerData'):
        return self._get_data('wgt')

    @wgt.setter
    def wgt(self: 'LayerData', value: LayerResult):
        self._set_data('wgt', value)

    @property
    def momenta(self: 'LayerData'):
        return self._get_data('momenta')

    @momenta.setter
    def momenta(self: 'LayerData', value: LayerResult):
        self._set_data('momenta', value)

    @property
    def f_real(self: 'LayerData'):
        return self._get_data('f_real')

    @f_real.setter
    def f_real(self: 'LayerData', value: LayerResult):
        self._set_data('f_real', value)

    @property
    def f_imag(self: 'LayerData'):
        return self._get_data('f_imag')

    @f_imag.setter
    def f_imag(self: 'LayerData', value: LayerResult):
        self._set_data('f_imag', value)

    @property
    def func_val(self: 'LayerData'):
        if len(self._pending_data) > 0:
            self.update('func_val_setter')
        return self._get_data('f_real') + 1j*self._get_data('f_imag')

    @func_val.setter
    def func_val(self: 'LayerData', value: NDArray[np.complexfloating]):
        if value is None:
            self._set_data('f_real', None)
            self._set_data('f_imag', None)
        if np.iscomplexobj(value):
            self._set_data('f_real', value.real)
            self._set_data('f_imag', value.imag)
        else:
            self._set_data('f_real', value)

    @property
    def continuous(self: 'LayerData'):
        return self._get_data('continuous')

    @continuous.setter
    def continuous(self: 'LayerData', value: LayerResult):
        self._set_data('continuous', value)

    @property
    def discrete(self: 'LayerData'):
        return self._get_data('discrete').astype(np.uint64)

    @discrete.setter
    def discrete(self: 'LayerData', value: LayerResult):
        self._set_data('discrete', value)

    def _to_layer_data(self: 'LayerData', value: LayerResult) -> NDArray:
        """
        Converts input to numpy NDArray of type self.dtype. Complex values must be passed
        separately as real and complex parts. Used in property setters.

        :param value: User data
        :type value: Tensor | NDArray | None
        :return:
        :rtype: NDArray
        """
        if value is None:
            return np.zeros(dtype=self.dtype, shape=(self.n_points, 0))

        match value:
            case list():
                output = np.array(value, dtype=self.dtype)
            case np.ndarray():
                output = value.astype(self.dtype)
            case _:
                raise ValueError(
                    "LayerData objects accept only numpy ndarray, list or None.")

        return output.reshape(self.n_points, -1)

    def _update_processing_times(self: 'LayerData', identifier: str = 'unspecified') -> None:
        processing_time = perf_counter() - self._timestamp
        if identifier in self._processing_times:
            self._processing_times[identifier] += processing_time
        else:
            self._processing_times[identifier] = processing_time
        self._timestamp = perf_counter()

    @_timer
    def _set_data(self: 'LayerData',
                  name: str,
                  value: LayerResult,) -> None:
        # state will only be updated when update is called
        self._pending_data[name] = self._to_layer_data(value)

    @_timer
    def _get_data(self: 'LayerData', name: str):
        if name in self._pending_data.keys():
            self.update(name+'_getter')

        idx = LayerData.POSITIONS[name]
        offset = self._structure[:idx].sum()
        dim = self._active_structure[idx]
        return self._data[:, offset:offset+dim]


@dataclass
class SinglePhaseResult:
    """
    Implements the Welford's online algorithm for mean and error calculation, which allows for numerically stable updates of the mean and error as new data points are added. This is particularly useful in the context of Monte Carlo integration, where we want to accumulate results from multiple samples without having to store all individual sample values.
    Attributes:
        mean (float): The current mean of the monte carlo estimate.
        error (float): The current error of the monte carlo estimate.
        rsd (float): The relative standard deviation of the monte carlo estimate.
        tvar (float): The relative time variance of the monte carlo estimate, calculated as RSD^2 * time per sample.
    """
    mean: float = 0
    _n: int = 0
    _total_time: float = 0
    _m2: float = 0  # Internal variable for error calculation

    @property
    def std(self: 'SinglePhaseResult') -> float:
        if self._n < 2:
            return 0.0  # Not enough data to calculate error
        return np.sqrt(self._m2 / (self._n - 1))

    @property
    def error(self: 'SinglePhaseResult') -> float:
        if self._n < 2:
            return 0.0
        return np.sqrt(self._m2 / (self._n - 1) / self._n)

    @property
    def rsd(self: 'SinglePhaseResult') -> float:
        if self.mean == 0:
            return 0.0
        return self.std / np.abs(self.mean)

    @property
    def tvar(self: 'SinglePhaseResult') -> float:
        if self._n == 0:
            return 0.0
        time_per_sample = self._total_time / self._n
        return self.rsd**2 * time_per_sample

    def combine_with(self: 'SinglePhaseResult', other: 'SinglePhaseResult') -> None:
        total_points = self._n + other._n
        if total_points == 0:
            return  # No data to combine
        delta = other.mean - self.mean
        self.mean += delta * other._n / total_points
        self._m2 += other._m2 + delta**2 * self._n * other._n / total_points
        self._n = total_points
        self._total_time += other._total_time

    def combine_with_stratified(self: 'SinglePhaseResult', other: 'SinglePhaseResult') -> None:
        """
        Combines this result with another SinglePhaseResult using stratified sampling combination rules. 
        Objects obtained this way should combine all stratified results at once, and be fully recalculated
        from strata each time a sample batch is added.
        """
        total_points = self._n + other._n
        if total_points == 0:
            return
        self.mean = (self.mean * self._n + other.mean * other._n) / total_points if total_points > 0 else 0
        # Achieves combined.error**2 = self.error**2 + other.error**2
        self._m2 = (
            (self._m2 / (self._n - 1) / self._n if self._n > 1 else 0.0)
            + (other._m2 / (other._n - 1) / other._n if other._n > 1 else 0.0)
        ) * (total_points - 1) * total_points if total_points > 1 else 0
        self._n = total_points
        self._total_time += other._total_time

    @classmethod
    def from_values(cls: 'SinglePhaseResult', values: NDArray, total_time: float = 0) -> 'SinglePhaseResult':
        n = values.size
        return cls(
            _n=n,
            _total_time=total_time,
            mean=np.mean(values).item() if n > 0 else 0.0,
            _m2=np.sum((values - np.mean(values))**2).item() if n > 1 else 0.0,
        )

    @classmethod
    def cat(cls: 'SinglePhaseResult', results: List['SinglePhaseResult']) -> 'SinglePhaseResult':
        combined = cls()
        for result in results:
            combined.combine_with(result)
        return combined


@dataclass
class Result:
    """
    Holds the results of the integration, including the number of points, total time taken, and the results for the real and 
    imaginary parts of the integral, as well as their absolute values. The results for each phase are stored in `SinglePhaseResult` 
    objects, which allow for stable updates of the mean and error as new data points are added. The Result class also provides methods 
    for combining results from different batches of samples, either using standard combination rules or stratified sampling combination rules.
    Attributes:
        n_points (int): The total number of points sampled.
        total_time (float): The total time taken for the integration.
        real (SinglePhaseResult): The result calculated on the real part of the integrand.
        imag (SinglePhaseResult): The result calculated on the imaginary part of the integrand.
        abs_real (SinglePhaseResult): The result calculated on the absolute value of the real part of the integrand.
        abs_imag (SinglePhaseResult): The result calculated on the absolute value of the imaginary part of the integrand.
    """
    n_points: int = 0
    total_time: float = 0
    real: SinglePhaseResult = field(default_factory=SinglePhaseResult)
    imag: SinglePhaseResult = field(default_factory=SinglePhaseResult)
    abs_real: SinglePhaseResult = field(default_factory=SinglePhaseResult)
    abs_imag: SinglePhaseResult = field(default_factory=SinglePhaseResult)

    def combine_with(self: 'Result', other: 'Result') -> None:
        self.n_points += other.n_points
        self.total_time += other.total_time
        self.real.combine_with(other.real)
        self.imag.combine_with(other.imag)
        self.abs_real.combine_with(other.abs_real)
        self.abs_imag.combine_with(other.abs_imag)

    def combine_with_stratified(self: 'Result', other: 'Result') -> None:
        self.n_points += other.n_points
        self.total_time += other.total_time
        self.real.combine_with_stratified(other.real)
        self.imag.combine_with_stratified(other.imag)
        self.abs_real.combine_with_stratified(other.abs_real)
        self.abs_imag.combine_with_stratified(other.abs_imag)

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'Result':
        n = kwargs.get('n_points', 0)
        total_time = kwargs.get('total_time', 0)
        return cls(
            n_points=n,
            total_time=total_time,
            real=SinglePhaseResult(
                mean=kwargs.get('real_mean', 0),
                _n=n,
                _total_time=total_time,
                _m2=kwargs.get('real_error', 0)**2 * n*(n-1) if n > 1 else 0
            ),
            imag=SinglePhaseResult(
                mean=kwargs.get('imag_mean', 0),
                _n=n,
                _total_time=total_time,
                _m2=kwargs.get('imag_error', 0)**2 * n*(n-1) if n > 1 else 0
            ),
            abs_real=SinglePhaseResult(
                mean=kwargs.get('abs_real_mean', 0),
                _n=n,
                _total_time=total_time,
                _m2=kwargs.get('abs_real_error', 0)**2 * n*(n-1) if n > 1 else 0
            ),
            abs_imag=SinglePhaseResult(
                mean=kwargs.get('abs_imag_mean', 0),
                _n=n,
                _total_time=total_time,
                _m2=kwargs.get('abs_imag_error', 0)**2 * n*(n-1) if n > 1 else 0
            ),
        )

    @classmethod
    def from_values(cls, real_values: NDArray, imag_values: NDArray, total_time: float = 0) -> 'Result':
        n_points = real_values.size
        assert n_points == imag_values.size, "Real and imaginary values must have the same number of points."
        return cls(
            n_points=n_points,
            total_time=total_time,
            real=SinglePhaseResult.from_values(real_values, total_time),
            imag=SinglePhaseResult.from_values(imag_values, total_time),
            abs_real=SinglePhaseResult.from_values(np.abs(real_values), total_time),
            abs_imag=SinglePhaseResult.from_values(np.abs(imag_values), total_time),
        )

    @classmethod
    def cat(cls, results: List['Result']) -> 'Result':
        combined = cls()
        for result in results:
            combined.combine_with(result)
        return combined

    @classmethod
    def from_legacy(cls, legacy_result: 'Result') -> 'Result':
        return cls.from_kwargs(
            n_points=legacy_result.n_points,
            total_time=legacy_result.total_time,
            real_mean=legacy_result.real_mean,
            real_error=legacy_result.real_error,
            imag_mean=legacy_result.imag_mean,
            imag_error=legacy_result.imag_error,
            abs_real_mean=legacy_result.abs_real_mean,
            abs_real_error=legacy_result.abs_real_error,
            abs_imag_mean=legacy_result.abs_imag_mean,
            abs_imag_error=legacy_result.abs_imag_error,
        )

    def to_legacy(self: 'Result') -> 'IntegrationResult':
        return IntegrationResult(
            n_points=self.n_points,
            total_time=self.total_time,
            real_mean=self.real.mean,
            real_error=self.real.error,
            imag_mean=self.imag.mean,
            imag_error=self.imag.error,
            abs_real_mean=self.abs_real.mean,
            abs_real_error=self.abs_real.error,
            abs_imag_mean=self.abs_imag.mean,
            abs_imag_error=self.abs_imag.error,
            real_rsd=self.real.rsd,
            imag_rsd=self.imag.rsd,
            abs_real_rsd=self.abs_real.rsd,
            abs_imag_rsd=self.abs_imag.rsd,
            real_tvar=self.real.tvar,
            imag_tvar=self.imag.tvar,
            abs_real_tvar=self.abs_real.tvar,
            abs_imag_tvar=self.abs_imag.tvar
        )


@dataclass
class IntegrationResult:
    """
    Legacy class, only used as a container now. Should not be used, since the new `Result` class 
    has proper implementation of running error calculation, and is more robust to updates. 
    """
    n_points: int = 0
    real_mean: float = 0
    real_error: float = 0
    imag_mean: float = 0
    imag_error: float = 0
    abs_real_mean: float = 0
    abs_real_error: float = 0
    abs_imag_mean: float = 0
    abs_imag_error: float = 0
    real_rsd: float = 0
    imag_rsd: float = 0
    abs_real_rsd: float = 0
    abs_imag_rsd: float = 0
    real_tvar: float = 0
    imag_tvar: float = 0
    abs_real_tvar: float = 0
    abs_imag_tvar: float = 0
    total_time: float = 0

    def __post_init__(self: 'IntegrationResult'):
        """Calculates derived observables such as RSD and time per sample."""
        time_per_sample = 0
        # No harm in recalculating these
        if self.n_points > 0:
            time_per_sample = self.total_time / self.n_points
            if self.real_mean:
                self.real_rsd = self._rsd(self.real_mean, self.real_error, self.n_points)
            if self.imag_mean:
                self.imag_rsd = self._rsd(self.imag_mean, self.imag_error, self.n_points)
            if self.abs_real_mean:
                self.abs_real_rsd = self._rsd(
                    self.abs_real_mean, self.abs_real_error, self.n_points)
            if self.abs_imag_mean:
                self.abs_imag_rsd = self._rsd(
                    self.abs_imag_mean, self.abs_imag_error, self.n_points)

        # Don't want to overwrite these in case total_time is not provided
        if time_per_sample > 0:
            self.real_tvar = self.real_rsd**2 * time_per_sample
            self.imag_tvar = self.imag_rsd**2 * time_per_sample
            self.abs_real_tvar = self.abs_real_rsd**2 * time_per_sample
            self.abs_imag_tvar = self.abs_imag_rsd**2 * time_per_sample

    def combine_with(self: 'IntegrationResult', other: 'IntegrationResult') -> None:
        def cc(s_c: float, o_c: float) -> float:
            return self._combine_mean(self.n_points, other.n_points, s_c, o_c)

        def ce(s_e: float, o_e: float) -> float:
            return self._combine_error(self.n_points, other.n_points, s_e, o_e)

        self.real_mean = cc(self.real_mean, other.real_mean)
        self.imag_mean = cc(self.imag_mean, other.imag_mean)
        self.abs_real_mean = cc(self.abs_real_mean, other.abs_real_mean)
        self.abs_imag_mean = cc(self.abs_imag_mean, other.abs_imag_mean)
        self.real_error = ce(self.real_error, other.real_error)
        self.imag_error = ce(self.imag_error, other.imag_error)
        self.abs_real_error = ce(self.abs_real_error, other.abs_real_error)
        self.abs_imag_error = ce(self.abs_imag_error, other.abs_imag_error)

        self.n_points += other.n_points
        self.total_time += other.total_time
        self.__post_init__()  # Recalculate derived observables

    def str_report(self: 'IntegrationResult') -> str:
        return f"""Real: {
            self.real_mean: .5f}  ± {
            self.real_error: .5f}, Imag: {
            self.imag_mean: .5f}  ± {
            self.imag_error: .5f} """

    @staticmethod
    def _rsd(mean: float, err: float, n_points: int) -> float:
        if mean == 0:
            return 0
        return err / np.abs(mean) * np.sqrt(n_points)

    @staticmethod
    def _combine_mean(s_n: float, o_n: float, s_c: float, o_c: float) -> float:
        return (s_n * s_c + o_n * o_c) / (s_n + o_n)

    @staticmethod
    def _combine_error(s_n: float, o_n: float, s_e: float, o_e: float) -> float:
        return np.sqrt(s_n**2 * s_e**2 + o_n**2 * o_e**2) / (s_n + o_n)
