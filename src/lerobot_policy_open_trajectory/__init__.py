"""Open Trajectory policy plugin for LeRobot."""

try:
    import lerobot as _lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by installation checks
    raise ImportError(
        "Open Trajectory requires LeRobot. "
        "Install lerobot[dataset]>=0.6.1,<0.7 before importing this package."
    ) from exc

from .configuration_open_trajectory import OpenTrajectoryConfig
from .configuration_open_trajectory_act import OpenTrajectoryACTConfig

__all__ = ["OpenTrajectoryACTConfig", "OpenTrajectoryConfig"]
