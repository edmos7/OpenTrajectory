"""Open Trajectory policy plugin for LeRobot."""

try:
    import lerobot as _lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by installation checks
    raise ImportError(
        "Open Trajectory requires LeRobot. Install lerobot>=0.6.1 before importing this package."
    ) from exc

from .configuration_open_trajectory import OpenTrajectoryConfig

__all__ = ["OpenTrajectoryConfig"]
