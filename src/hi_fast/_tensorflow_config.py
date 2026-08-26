"""Import TensorFlow without its non-actionable startup diagnostics."""
from contextlib import contextmanager
import importlib
import os
import sys
import tempfile


# Keep the conventional TensorFlow filter as a user-overridable default. Newer
# TensorFlow releases emit some Abseil messages before this filter is active,
# so the import itself is also handled below.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


@contextmanager
def _suppress_startup_logs():
    """Capture native stderr while TensorFlow initializes.

    Redirecting ``sys.stderr`` is insufficient because TensorFlow's C++
    runtime writes directly to file descriptor 2. If importing TensorFlow
    fails, replay the captured diagnostics after restoring stderr so useful
    error information is never lost.
    """
    original_stderr = os.dup(2)
    failed = False
    with tempfile.TemporaryFile(mode='w+b') as captured:
        try:
            os.dup2(captured.fileno(), 2)
            yield
        except BaseException:
            failed = True
            raise
        finally:
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
            if failed:
                captured.seek(0)
                sys.stderr.write(captured.read().decode(errors='replace'))
                sys.stderr.flush()


with _suppress_startup_logs():
    tf = importlib.import_module('tensorflow')
    keras = tf.keras
