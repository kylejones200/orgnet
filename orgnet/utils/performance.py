"""Performance optimization utilities using numba, polars, and parquet."""

import numpy as np
from typing import Optional, Union, Callable, Dict, TYPE_CHECKING
import pandas as pd
import hashlib
from pathlib import Path

if TYPE_CHECKING:
    # Only import polars types for type checking
    try:
        import polars as pl
    except ImportError:
        pl = None

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    Parallel = None
    delayed = None

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

try:
    import pyarrow.parquet as pq
    import pyarrow as pa

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pq = None
    pa = None


@jit(nopython=True, cache=True)
def compute_constraint_numba(neighbor_indices, weights, total_weight):
    """
    Compute constraint score using numba for performance.

    Args:
        neighbor_indices: Array of neighbor indices (0-based)
        weights: Array of edge weights
        total_weight: Total weight sum

    Returns:
        Constraint score
    """
    n = len(neighbor_indices)
    if n < 2 or total_weight == 0.0:
        return 1.0

    constraint = 0.0

    for i in range(n):
        p_ij = weights[i] / total_weight

        indirect = 0.0
        for k_idx in range(n):
            if k_idx == i:
                continue

            p_ik = weights[k_idx] / total_weight
            indirect += p_ik * p_ik

        constraint += (p_ij + indirect) ** 2

    return constraint


@jit(nopython=True, cache=True)
def cosine_similarity_batch(embeddings, query_embedding):
    """
    Compute cosine similarity between query and batch of embeddings.

    Args:
        embeddings: 2D array of embeddings (n_samples, n_features)
        query_embedding: 1D array query embedding

    Returns:
        Array of similarity scores
    """
    n_samples = embeddings.shape[0]
    similarities = np.zeros(n_samples)

    query_norm = np.sqrt(np.sum(query_embedding**2))

    for i in prange(n_samples):
        dot_product = np.sum(embeddings[i] * query_embedding)
        embedding_norm = np.sqrt(np.sum(embeddings[i] ** 2))
        similarities[i] = dot_product / (query_norm * embedding_norm + 1e-10)

    return similarities


@jit(nopython=True, cache=True)
def sentiment_score_numba(words_array, positive_set, negative_set, negators_set, intensifiers_set):
    """
    Compute sentiment score using numba.

    Args:
        words_array: Array of word indices (mapped from strings)
        positive_set: Set of positive word indices
        negative_set: Set of negative word indices
        negators_set: Set of negator word indices
        intensifiers_set: Set of intensifier word indices

    Returns:
        Sentiment score
    """
    n_words = len(words_array)
    if n_words == 0:
        return 0.0

    positive_count = 0.0
    negative_count = 0.0

    for i in range(n_words):
        word = words_array[i]
        is_negated = False
        is_intensified = False

        for j in range(max(0, i - 3), i):
            if words_array[j] in negators_set:
                is_negated = True
                break

        for j in range(max(0, i - 2), i):
            if words_array[j] in intensifiers_set:
                is_intensified = True
                break

        multiplier = 2.0 if is_intensified else 1.0

        if word in positive_set:
            if is_negated:
                negative_count += multiplier
            else:
                positive_count += multiplier
        elif word in negative_set:
            if is_negated:
                positive_count += multiplier
            else:
                negative_count += multiplier

    return (positive_count - negative_count) / n_words


def pandas_to_polars(df: pd.DataFrame) -> Optional["pl.DataFrame"]:
    """Convert pandas DataFrame to polars DataFrame."""
    if not POLARS_AVAILABLE:
        return None
    try:
        return pl.from_pandas(df)
    except Exception:
        return None


def polars_to_pandas(df_pl: "pl.DataFrame") -> pd.DataFrame:
    """Convert polars DataFrame to pandas DataFrame."""
    if not POLARS_AVAILABLE:
        raise ImportError("polars not available")
    return df_pl.to_pandas()


def save_parquet(df: Union[pd.DataFrame, "pl.DataFrame"], path: str) -> bool:
    """
    Save DataFrame to parquet format.

    Args:
        df: pandas or polars DataFrame
        path: Output file path

    Returns:
        True if successful
    """
    if not PARQUET_AVAILABLE:
        return False

    try:
        if POLARS_AVAILABLE and pl is not None and isinstance(df, pl.DataFrame):
            df.write_parquet(path)
        else:
            df.to_parquet(path, engine="pyarrow", index=False)
        return True
    except Exception:
        return False


def load_parquet(
    path: str, use_polars: bool = False
) -> Union[pd.DataFrame, Optional["pl.DataFrame"]]:
    """
    Load DataFrame from parquet format.

    Args:
        path: Input file path
        use_polars: If True, return polars DataFrame (if available)

    Returns:
        pandas or polars DataFrame
    """
    if not PARQUET_AVAILABLE:
        raise ImportError("parquet support not available")

    if use_polars and POLARS_AVAILABLE:
        return pl.read_parquet(path)
    else:
        return pd.read_parquet(path, engine="pyarrow")


@jit(nopython=True, cache=True)
def z_score_numba(values, mean, std):
    """Compute z-scores using numba."""
    n = len(values)
    z_scores = np.zeros(n)
    if std > 1e-10:
        for i in prange(n):
            z_scores[i] = abs((values[i] - mean) / std)
    return z_scores


@jit(nopython=True, cache=True)
def rolling_mean_numba(values, window):
    """Compute rolling mean using numba."""
    n = len(values)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start : i + 1])
    return result


@jit(nopython=True, cache=True)
def compute_edge_weights_numba(interaction_counts, time_decay, base_weight):
    """Compute edge weights with time decay using numba."""
    n = len(interaction_counts)
    weights = np.zeros(n)
    for i in prange(n):
        weights[i] = base_weight * interaction_counts[i] * (1.0 - time_decay[i])
    return weights


def optimize_dataframe_ops(df: pd.DataFrame, operation: str = "auto") -> pd.DataFrame:
    """
    Optimize DataFrame operations using polars when beneficial.

    Args:
        df: pandas DataFrame
        operation: Operation type ('auto', 'groupby', 'join', 'filter')

    Returns:
        Optimized pandas DataFrame
    """
    if not POLARS_AVAILABLE or len(df) < 10000:
        return df

    try:
        df_pl = pl.from_pandas(df)

        if operation == "groupby":
            df_pl = df_pl.lazy()
        elif operation == "join":
            df_pl = df_pl.lazy()
        elif operation == "filter":
            df_pl = df_pl.lazy()

        return df_pl.collect().to_pandas()
    except Exception:
        return df


def polars_groupby(df: pd.DataFrame, by: Union[str, list], agg: Dict) -> pd.DataFrame:
    """
    Perform groupby operation using polars for better performance.

    Args:
        df: pandas DataFrame
        by: Column(s) to group by
        agg: Aggregation dictionary

    Returns:
        pandas DataFrame with grouped results
    """
    if not POLARS_AVAILABLE or len(df) < 10000:
        return df.groupby(by).agg(agg).reset_index()

    try:
        df_pl = pl.from_pandas(df)
        if isinstance(by, str):
            by = [by]

        polars_agg = []
        for col, func in agg.items():
            if isinstance(func, str):
                if func == "mean":
                    polars_agg.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif func == "std":
                    polars_agg.append(pl.col(col).std().alias(f"{col}_std"))
                elif func == "sum":
                    polars_agg.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif func == "count":
                    polars_agg.append(pl.col(col).count().alias(f"{col}_count"))
            elif isinstance(func, list):
                for f in func:
                    if f == "mean":
                        polars_agg.append(pl.col(col).mean().alias(f"{col}_mean"))
                    elif f == "std":
                        polars_agg.append(pl.col(col).std().alias(f"{col}_std"))

        result = df_pl.group_by(by).agg(polars_agg)
        return result.to_pandas()
    except Exception:
        return df.groupby(by).agg(agg).reset_index()


def polars_join(
    left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "inner"
) -> pd.DataFrame:
    """
    Perform join operation using polars for better performance.

    Args:
        left: Left pandas DataFrame
        right: Right pandas DataFrame
        on: Column to join on
        how: Join type ('inner', 'left', 'right', 'outer')

    Returns:
        pandas DataFrame with joined results
    """
    if not POLARS_AVAILABLE or len(left) < 10000 or len(right) < 10000:
        return left.merge(right, on=on, how=how)

    try:
        left_pl = pl.from_pandas(left)
        right_pl = pl.from_pandas(right)

        result = left_pl.join(right_pl, on=on, how=how)
        return result.to_pandas()
    except Exception:
        return left.merge(right, on=on, how=how)


class ParquetCache:
    """Parquet-based caching for expensive computations."""

    def __init__(self, cache_dir: str = ".cache/orgnet"):
        """
        Initialize parquet cache.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function name and arguments."""
        key_data = f"{func_name}_{args}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.parquet"

    def get(self, func_name: str, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        Get cached result if available.

        Args:
            func_name: Name of the function
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Cached DataFrame or None
        """
        if not PARQUET_AVAILABLE:
            return None

        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                return load_parquet(str(cache_path))
            except Exception:
                return None
        return None

    def set(self, func_name: str, result: pd.DataFrame, *args, **kwargs) -> bool:
        """
        Cache computation result.

        Args:
            func_name: Name of the function
            result: DataFrame result to cache
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            True if successful
        """
        if not PARQUET_AVAILABLE:
            return False

        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        try:
            return save_parquet(result, str(cache_path))
        except Exception:
            return False

    def clear(self, func_name: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            func_name: If provided, clear only this function's cache
        """
        if func_name:
            pattern = f"*{func_name}*.parquet"
            for f in self.cache_dir.glob(pattern):
                f.unlink()
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()


def cached_computation(cache: ParquetCache, func_name: str):
    """
    Decorator for caching expensive computations.

    Args:
        cache: ParquetCache instance
        func_name: Name of the function for caching
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            cached_result = cache.get(func_name, *args, **kwargs)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                cache.set(func_name, result, *args, **kwargs)
            return result

        return wrapper

    return decorator


def parallel_map(func: Callable, items: list, n_jobs: int = -1) -> list:
    """
    Apply function to items in parallel.

    Args:
        func: Function to apply
        items: List of items to process
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        List of results
    """
    if not JOBLIB_AVAILABLE or len(items) < 100:
        return [func(item) for item in items]

    try:
        return Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in items)
    except Exception:
        return [func(item) for item in items]


def parallel_groupby_apply(
    df: pd.DataFrame, by: str, func: Callable, n_jobs: int = -1
) -> pd.DataFrame:
    """
    Apply function to groups in parallel.

    Args:
        df: pandas DataFrame
        by: Column to group by
        func: Function to apply to each group
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame with results
    """
    if not JOBLIB_AVAILABLE or len(df) < 1000:
        return df.groupby(by).apply(func).reset_index(drop=True)

    try:
        groups = [group for _, group in df.groupby(by)]
        results = Parallel(n_jobs=n_jobs)(delayed(func)(group) for group in groups)
        return pd.concat(results, ignore_index=True)
    except Exception:
        return df.groupby(by).apply(func).reset_index(drop=True)
