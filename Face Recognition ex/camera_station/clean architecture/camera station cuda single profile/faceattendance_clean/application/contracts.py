"""Application-layer ports used by presentation and infrastructure adapters.

The legacy station still contains the heavy OpenCV/Tkinter implementation, but
new code should depend on these protocols instead of importing UI or storage
classes directly.  This keeps the migration toward clean architecture gradual and
safe.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Protocol, Sequence

import numpy as np


class KnownFaceRepository(Protocol):
    def load_all(self) -> Dict[str, List[np.ndarray]]:
        ...

    def save_person(self, label: str, descriptors: Sequence[np.ndarray]) -> None:
        ...

    def delete_all(self) -> None:
        ...


class UnknownFaceRepository(Protocol):
    def export_package(self, destination_zip: str) -> str:
        ...

    def import_package(self, source_zip: str) -> int:
        ...


class RecognitionWorkerPort(Protocol):
    def start(self) -> None:
        ...

    def request_stop(self) -> None:
        ...

    def update_controls(self, **kwargs) -> None:
        ...

    def register_known(self, label: str, descriptors: Sequence[np.ndarray]) -> None:
        ...

    def clear_known_database(self, disable_reload: bool = True) -> None:
        ...
