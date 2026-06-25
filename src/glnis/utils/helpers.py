# type: ignore
import math
from enum import StrEnum
from typing import Dict, Sequence, List, Any, Iterable
from pathlib import Path


class Colour(StrEnum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def error_fmter(value: float, error: float, prec_error: int = 2) -> str:
    """
    Format a value and its error in scientific notation with a given number of significant digits for the error.

    Examples:
        value = 1234.5678, error = 111.11, prec_error = 2 -> "1.23(11)e+04"
        value = 12.345, error = 111.111, prec_error = 1 -> "1.2(11.1)e+01"
        value = 0.0123, error = 0.001234, prec_error = 3 -> "1.230(123)e-02"
    """
    if error < 0:
        raise ValueError("Error must be positive.")
    prec_error = max(1, prec_error)

    if value == 0:
        log10val = 0
    else:
        log10val = math.floor(math.log10(abs(value)))

    exp10val = 10.0**log10val

    # Normalize both value and error to the same order of magnitude
    val_norm = value / exp10val
    err_norm = error / exp10val

    # Set prec: the significant number of digits such that prec_error number
    # of significant digits are shown for the error
    if error == 0:
        val_str = f"{val_norm:.{prec_error}f}"
        return f"{val_str}(0)e{log10val:+03d}"

    log10err_norm = math.floor(math.log10(err_norm))

    if log10err_norm >= 0:
        prec = prec_error
    else:
        prec = prec_error - log10err_norm - 1

    # Get digits without scientific notation
    val_str = f"{val_norm:.{prec}f}"
    if log10err_norm >= 0:
        err_str = f"{err_norm:.{prec}f}"
    else:
        err_str = f"{err_norm:.{prec}e}".replace(".", "")[:prec_error]
    # I don't think this can happen since error>0, but if the error is somehow rounded
    # down to zero, err_str will be empty and we default to
    if not err_str:
        err_str = '0' * prec_error

    return f"{val_str}({err_str})e{log10val:+03d}"


def time_fmter(seconds: float, prefix: str = "", n_digits: int = 3) -> str:
    """
    Format a time duration in seconds into a human-readable string with appropriate units.

    Examples:
        digits = 3
        seconds = 0.000001 -> "1.00 µs"
        seconds = 0.01 -> "10.0 ms"
        seconds = 1 -> "1.00 s"
        seconds = 120 -> "2.00 min"
        seconds = 4000 -> "1.11 h"
    """
    number: float = 0
    suffix: str = " " + prefix
    if seconds < 1e-6:
        suffix += "ns"
        number = seconds * 1e9
    elif seconds < 1e-3:
        suffix += "µs"
        number = seconds * 1e6
    elif seconds < 1:
        suffix += "ms"
        number = seconds * 1e3
    elif seconds < 60:
        suffix += "s"
        number = seconds
    elif seconds < 3600:
        suffix += "min"
        number = seconds / 60
    else:
        suffix += "h"
        number = seconds / 3600

    prec = n_digits - 1 - math.floor(math.log10(abs(number))) if number != 0 else n_digits
    number_str = f"{number:.{prec}f}"

    return number_str + suffix


def chunks(ary: Sequence, n_chunks: int) -> Iterable[Sequence]:
    """
    Like numpy.array_split, but works for all sequences and returns an iterator.
    """
    ln_ary = len(ary)
    if n_chunks > ln_ary or n_chunks < 1:
        raise ValueError(
            "the number of chunks should be at least 1, and at most len(ary)")
    n_long = ln_ary % n_chunks
    ln_long = ln_ary // n_chunks + 1
    total_long = n_long*ln_long
    ln_short = ln_ary // n_chunks

    for start in range(0, total_long, ln_long):
        yield ary[start:start+ln_long]
    for start in range(total_long, ln_ary, ln_short):
        yield ary[start:start+ln_short]


def overwrite_settings(orig_dict: Dict[str, Any], new_dict: Dict[str, Any],
                       always_overwrite: List[str] = None,
                       ) -> Dict[str, Any]:
    """
    Used to Overwrite the default settings with the specified settings file.
    """
    always_overwrite = always_overwrite or []
    if "overwrite" in new_dict:
        orig_dict["overwrite"] = new_dict["overwrite"]
        overwrite = new_dict["overwrite"]
        for joined_keys, value in overwrite.items():
            keys = joined_keys.split(".")
            d = orig_dict
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

    for force in always_overwrite:
        if force in new_dict.keys():
            orig_dict[force] = new_dict[force]

    for key, val in new_dict.items():
        if isinstance(val, Dict) and key in orig_dict.keys():
            tmp = overwrite_settings(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = val

    return orig_dict


def shell_print(*lines: str, prefix="| >"):
    for line in lines:
        for ln in line.split("\n"):
            print(prefix, ln)


def verify_path(path: str, suffix: str = None, _levels_to_root: int = 3) -> Path:
    """
    Verify that the path is valid and exists. If the path is relative, it will be converted to an 
    absolute path based on the current working directory and the levels_to_root.

    Args:
        path (str): The path to verify.
        suffix (str): The suffix to force onto the file, if it doesn't already have it.
        _levels_to_root (int): The number of levels to go up from the directory of the helpers file to reach the root of the project. 
        Default is 3. 
    Returns:
        Path: The verified and converted path.
    """
    path: Path = Path(path)
    if not path.suffix and suffix is not None:
        path = path.with_suffix(suffix)
    if not path.is_absolute():
        PROJECT_ROOT = Path(__file__).parents[_levels_to_root]
        path = Path(PROJECT_ROOT, path)
    if not path.exists():
        raise FileNotFoundError(
            f"File at '{path}' does not exist. Path must be either absolute, or relative to the glnis root folder.")
    return path


def _open_fd_count() -> int | None:
    proc_fd_path = Path('/proc/self/fd')
    if not proc_fd_path.exists():
        return None
    try:
        return len(list(proc_fd_path.iterdir()))
    except Exception:
        return None


def _fd_limit() -> int | None:
    try:
        import resource
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:
        return None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except:
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def set_nice_plotting_style():
    import matplotlib.pyplot as plt
    # Set up publication-style parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 3
