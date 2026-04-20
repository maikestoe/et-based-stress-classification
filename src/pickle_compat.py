"""Compatibility helpers for loading historical pandas pickle files."""

import sys
import types


def install_pandas_pickle_compat() -> None:
    """Register aliases needed by pickles created with older pandas versions."""
    try:
        import pandas as pd
    except ImportError:
        return

    module_name = "pandas.core.indexes.numeric"
    if module_name in sys.modules:
        return

    numeric_module = types.ModuleType(module_name)
    numeric_module.Int64Index = pd.Index
    numeric_module.UInt64Index = pd.Index
    numeric_module.Float64Index = pd.Index
    sys.modules[module_name] = numeric_module
