"""Private, coverage-aware cache for the latest HiCLASS computation."""

from contextlib import contextmanager
import threading

import hiclassy
import numpy as np


class HiClassCache:
    """Retain one computed HiCLASS instance for compatible requests."""

    _LIMIT_KEYS = (
        'P_k_max_h/Mpc',
        'P_k_max_1/Mpc',
        'z_max_pk',
        'l_max_scalars',
    )

    def __init__(self):
        self._lock = threading.RLock()
        self._cosmo = None
        self._base_key = None
        self._params = None
        self._coverage = None
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _freeze(value):
        """Convert nested parameter values into a deterministic cache key."""
        if isinstance(value, dict):
            return tuple(sorted(
                (key, HiClassCache._freeze(item))
                for key, item in value.items()))
        if isinstance(value, np.ndarray):
            return ('array', value.dtype.str, value.shape,
                    tuple(value.ravel().tolist()))
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return tuple(HiClassCache._freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(HiClassCache._freeze(item) for item in value))
        return value

    @staticmethod
    def _outputs(value):
        """Normalize CLASS's comma-separated output setting."""
        if value is None:
            return frozenset()
        if isinstance(value, str):
            return frozenset(
                item.strip() for item in value.split(',') if item.strip())
        return frozenset(value)

    @staticmethod
    def _limit(value):
        """Normalize an optional monotonic coverage limit."""
        if value is None:
            return None
        return float(value)

    @classmethod
    def _request_parts(cls, params):
        """Separate physical settings from actual computed coverage."""
        params = params.copy()
        coverage = {
            'output': cls._outputs(params.pop('output', None)),
            'limits': {
                key: cls._limit(params.pop(key, None))
                for key in cls._LIMIT_KEYS
            },
        }
        return cls._freeze(params), coverage

    @classmethod
    def _requested_coverage(cls, params, requirements):
        """Return only the coverage required by the calling observable."""
        if requirements is None:
            return cls._request_parts(params)[1]
        return {
            'output': cls._outputs(requirements.get('output')),
            'limits': {
                key: cls._limit(requirements.get(key))
                for key in cls._LIMIT_KEYS
            },
        }

    @classmethod
    def _covers(cls, cached, requested):
        """Return whether cached outputs and ranges cover a request."""
        if not requested['output'].issubset(cached['output']):
            return False
        for key in cls._LIMIT_KEYS:
            requested_value = requested['limits'][key]
            cached_value = cached['limits'][key]
            if requested_value is None:
                continue
            if cached_value is None or cached_value < requested_value:
                return False
        return True

    @classmethod
    def _merge_params(cls, cached_params, requested_params):
        """Return a request covering both compatible computations."""
        merged = requested_params.copy()
        outputs = cls._outputs(cached_params.get('output'))
        outputs |= cls._outputs(requested_params.get('output'))
        if outputs:
            merged['output'] = ', '.join(sorted(outputs))
        for key in cls._LIMIT_KEYS:
            cached = cls._limit(cached_params.get(key))
            requested = cls._limit(requested_params.get(key))
            values = [value for value in (cached, requested)
                      if value is not None]
            if values:
                merged[key] = max(values)
        return merged

    @staticmethod
    def _cleanup(cosmo):
        """Release native CLASS allocations without masking prior errors."""
        if cosmo is None:
            return
        try:
            cosmo.struct_cleanup()
        except Exception:
            pass
        try:
            cosmo.empty()
        except Exception:
            pass

    @staticmethod
    def _compute(params):
        """Create and compute a new HiCLASS instance."""
        cosmo = hiclassy.HiClass()
        try:
            cosmo.set(params)
            cosmo.compute()
        except Exception:
            HiClassCache._cleanup(cosmo)
            raise
        return cosmo

    def _get_or_compute(self, params, requirements=None):
        """Return a reusable instance or replace it with a new computation."""
        requested_params = params.copy()
        base_key, _ = self._request_parts(requested_params)
        requested_coverage = self._requested_coverage(
            requested_params, requirements)
        if (self._cosmo is not None
                and base_key == self._base_key
                and self._covers(self._coverage, requested_coverage)):
            self._hits += 1
            return self._cosmo

        compute_params = requested_params
        if self._cosmo is not None and base_key == self._base_key:
            compute_params = self._merge_params(self._params, requested_params)
        new_cosmo = self._compute(compute_params)
        new_key, new_coverage = self._request_parts(compute_params)
        old_cosmo = self._cosmo
        self._cosmo = new_cosmo
        self._base_key = new_key
        self._params = compute_params.copy()
        self._coverage = new_coverage
        self._misses += 1
        self._cleanup(old_cosmo)
        return self._cosmo

    @contextmanager
    def use(self, params, requirements=None):
        """Yield a computed instance while protecting its native state."""
        with self._lock:
            yield self._get_or_compute(params, requirements=requirements)

    def clear(self):
        """Clear the cached computation and release native allocations."""
        with self._lock:
            self._cleanup(self._cosmo)
            self._cosmo = None
            self._base_key = None
            self._params = None
            self._coverage = None

    def info(self):
        """Return private diagnostics for tests and profiling."""
        with self._lock:
            coverage = self._coverage
            return {
                'computed': self._cosmo is not None,
                'hits': self._hits,
                'misses': self._misses,
                'outputs': ([] if coverage is None
                            else sorted(coverage['output'])),
                'limits': ({} if coverage is None
                           else coverage['limits'].copy()),
            }

    def __del__(self):
        try:
            self.clear()
        except Exception:
            pass
