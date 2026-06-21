"""Infrastructure adapters around the existing OpenCV/Moodle implementation."""
from __future__ import annotations

from typing import Any


class LegacyStationModuleAdapter:
    """Thin adapter over ``moodle_yunet_sface_station``.

    Keeping this wrapper makes it clear where the current external framework
    boundary is.  The worker can later be moved behind this adapter without
    changing the UI layer.
    """

    def __init__(self, station_module: Any):
        self._station = station_module

    @property
    def station(self) -> Any:
        return self._station

    def apply_profile(self, profile: str, source: str) -> None:
        self._station.apply_algorithm_profile(profile, source=source)

    def build_known_feature_index(self, known_features):
        return self._station.build_known_feature_index(known_features)
