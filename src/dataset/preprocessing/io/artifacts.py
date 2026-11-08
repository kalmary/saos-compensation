"""Atomic persistence of trusted local fitted preprocessing artifacts."""

import pickle
from pathlib import Path

import sklearn

from ...storage.files import write_bytes

VERSION = 3
DEFAULT_PATH = Path(__file__).resolve().parents[4] / "checkpoints" / "preprocessing" / "feature_preprocessor.pkl"

DEFAULT_TARGET_PATH = DEFAULT_PATH.with_name("target_preprocessor.pkl")


def save_state(path, state):
    """Persist fitted state and the dependency version required to restore it.

    args:
        path: Artifact destination; None selects checkpoints/preprocessing/feature_preprocessor.pkl.
        state: Fitted preprocessing instance, including settings and column schema.

    returns:
        None.
    """
    payload = {"version": VERSION, "sklearn": sklearn.__version__, "state": state}
    write_bytes(DEFAULT_PATH if path is None else Path(path), pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def load_state(path):
    """Load a trusted pickle artifact and reject incompatible serialization versions.

    args:
        path: Trusted local artifact or None for the default checkpoint path.
            Pickle files may execute code when loaded.

    returns:
        Restored fitted preprocessing state.
    """
    payload = pickle.loads((DEFAULT_PATH if path is None else Path(path)).read_bytes())
    if not isinstance(payload, dict) or payload.get("version") != VERSION or payload.get("sklearn") != sklearn.__version__:
        raise ValueError("Incompatible preprocessing artifact version")
    return payload["state"]
