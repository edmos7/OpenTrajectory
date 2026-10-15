"""Gripper event detection: grasp/release peaks and sharpest-drop heuristic."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def detect_action_spikes(
    signal,
    prominence_frac: float = 0.3,
    min_gap: int = 20,
) -> list[int]:
    """Detect frames at significant peaks and troughs of the action signal.

    Uses :func:`scipy.signal.find_peaks` on the signal itself (and its
    negative) to find local extrema whose prominence is at least
    ``prominence_frac * signal_range``.  For a gripper signal this yields
    one frame per *held* interaction state (e.g. the fully-closed trough)
    rather than at the two gradient transitions that bound each event.

    Parameters
    ----------
    signal : array-like
        1-D action time series.
    prominence_frac : float
        Minimum prominence as a fraction of the signal's total value range.
        Defaults to 0.3.
    min_gap : int
        Minimum distance in frames between returned extrema.  Passed
        directly to ``find_peaks`` as ``distance``.  Defaults to 20.

    Returns
    -------
    list[int]
        Sorted frame indices of detected extrema.
    """
    s = np.asarray(signal, dtype=float)
    sig_range = float(s.max() - s.min())
    if sig_range < 1e-6:
        return []

    prominence = prominence_frac * sig_range

    pos_peaks, _ = find_peaks(s, prominence=prominence, distance=min_gap)
    neg_peaks, _ = find_peaks(-s, prominence=prominence, distance=min_gap)

    return sorted(set(pos_peaks.tolist() + neg_peaks.tolist()))


def detect_grasp_release(
    target_positions,
    prominence: float = 1.0,
    width: int = 1,
) -> tuple[list[int], list[int]]:
    """
    Detect grasp (sharp drop) and release (sharp rise) events in a gripper
    position time series.

    Args:
        target_positions: 1-D array-like of gripper target positions over time.
        prominence:       Minimum gradient peak prominence.
        width:            Minimum gradient peak width in samples.

    Returns:
        (grasping_steps, releasing_steps) - lists of frame indices.
    """
    pos = np.asarray(target_positions, dtype=float)
    grad = np.diff(pos)
    grasp_idx, _ = find_peaks(-grad, prominence=prominence, width=width)
    release_idx, _ = find_peaks(grad, prominence=prominence, width=width)
    return list(grasp_idx + 1), list(release_idx + 1)


def get_grasp_release_pairs(
    target_positions,
    prominence: float = 10.0,
    width: int = 1,
    min_hold_frames: int = 10,
    max_pairs: int = 5,
    min_close_duration: int = 15,
) -> list[tuple[int, int]]:
    """
    Return all valid (grasp, release) pairs from a gripper position series.

    Constraints enforced:
        1. No release before the first grasp.
        2. Every grasp must be followed by a release (unpaired grasps are dropped).
        3. Grasp and release gradients must have opposite signs (gripper closes then
           opens), ensuring spurious signal peaks are rejected.
        4. The hold duration (release - grasp) must be at least ``min_hold_frames``
           samples.  Short flicks caused by accidental or involuntary gripper
           movements in poor-quality demonstrations are discarded.
        5. **Twitch filter** (symmetric): a grasp candidate is discarded if the
           signal rebounds past 50 % of the closing amplitude within
           ``min_close_duration`` frames (close-open twitch).  Symmetrically, a
           release candidate is discarded if the signal drops back below 50 % of
           the opening amplitude within ``min_close_duration`` frames (open-close
           twitch).  Both directions filter transient motions that do not represent
           real grasp or release events.
        6. At most ``max_pairs`` pairs are returned; the first (chronologically
           earliest) pairs are kept.

    Pairs are matched greedily in chronological order: each grasp is paired with
    the earliest subsequent release that has not already been claimed.

    Args:
        target_positions:  1-D array-like of gripper target positions over time.
        prominence:        Passed to detect_grasp_release.
        width:             Passed to detect_grasp_release.
        min_hold_frames:   Minimum number of frames between a grasp and its paired
                           release.  Pairs shorter than this are treated as spurious
                           and discarded.  Defaults to 10.
        max_pairs:         Maximum number of grasp-release pairs to return.
                           Defaults to 5.
        min_close_duration: Minimum number of frames the gripper must remain below
                           the 50 % rebound threshold after a close event for it to
                           count as a real grasp (not a twitch).  Defaults to 25.

    Returns:
        List of (grasp_step, release_step) tuples in chronological order, or [].
    """
    grasps, releases = detect_grasp_release(target_positions, prominence, width)
    grasps = sorted(grasps)
    releases = sorted(releases)
    pos = np.asarray(target_positions, dtype=float)

    # ── Twitch filter ─────────────────────────────────────────────────────────
    # Applied symmetrically to both grasps (close-open twitches) and releases
    # (open-close twitches).  A candidate is discarded when the signal rebounds
    # past the 50 % amplitude mark within min_close_duration frames, indicating
    # the motion was too brief to represent a real grasp or release.

    def _is_close_open_twitch(g_idx: int) -> bool:
        """Grasp candidate: did the gripper reopen too quickly after closing?"""
        lookback = min(g_idx, 15)
        pre_close = float(np.median(pos[g_idx - lookback : g_idx])) if lookback > 0 else float(pos[0])
        at_close = float(pos[g_idx])
        amplitude = pre_close - at_close  # positive when gripper closed
        if amplitude <= 0:
            return True  # no actual closing motion
        reopen_threshold = at_close + 0.5 * amplitude
        window_end = min(g_idx + min_close_duration, len(pos))
        return bool(np.any(pos[g_idx:window_end] > reopen_threshold))

    def _is_open_close_twitch(r_idx: int) -> bool:
        """Release candidate: did the gripper re-close too quickly after opening?"""
        lookback = min(r_idx, 15)
        pre_open = float(np.median(pos[r_idx - lookback : r_idx])) if lookback > 0 else float(pos[0])
        at_open = float(pos[r_idx])
        amplitude = at_open - pre_open  # positive when gripper opened
        if amplitude <= 0:
            return True  # no actual opening motion
        reclose_threshold = at_open - 0.5 * amplitude
        window_end = min(r_idx + min_close_duration, len(pos))
        return bool(np.any(pos[r_idx:window_end] < reclose_threshold))

    grasps = [g for g in grasps if not _is_close_open_twitch(g)]
    releases = [r for r in releases if not _is_open_close_twitch(r)]

    # Drop releases that precede the first grasp
    if grasps:
        releases = [r for r in releases if r > grasps[0]]

    pairs: list[tuple[int, int]] = []
    gi = ri = 0
    while gi < len(grasps) and ri < len(releases):
        grasp = grasps[gi]
        while ri < len(releases) and releases[ri] <= grasp:
            ri += 1
        if ri >= len(releases):
            break
        release = releases[ri]
        while gi + 1 < len(grasps) and grasps[gi + 1] < release:
            gi += 1
        pairs.append((grasps[gi], release))
        gi += 1
        ri += 1

    if not pairs:
        return []

    def _opposite_signs(g_idx: int, r_idx: int) -> bool:
        g_grad = pos[g_idx] - pos[g_idx - 1] if g_idx > 0 else 0.0
        r_grad = pos[r_idx + 1] - pos[r_idx] if r_idx < len(pos) - 1 else 0.0
        return g_grad != 0 and r_grad != 0 and np.sign(g_grad) != np.sign(r_grad)

    valid = [(g, r) for g, r in pairs if _opposite_signs(g, r) and (r - g) >= min_hold_frames]
    return valid[:max_pairs]


def find_sharpest_drop(
    action_seq: np.ndarray,
    smooth_window: int = 15,
    smooth_poly: int = 2,
    gradient_threshold: float = 1.0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the start of the longest contiguous interval in which the distance
    from the episode's initial pose decreases sharply.

    Args:
        action_seq:         (T, n_actions) array of joint actions.
        smooth_window:      Savitzky-Golay window length (must be odd).
        smooth_poly:        Savitzky-Golay polynomial order.
        gradient_threshold: Minimum rate of decrease to qualify as «sharp».

    Returns:
        (interval_start_idx, D_raw, D_smooth, gradient)
    """
    start_pose = action_seq[0]
    D_raw = np.linalg.norm(action_seq - start_pose, axis=1)
    win = smooth_window if len(D_raw) > smooth_window else max(3, (len(D_raw) // 5) | 1)
    D_smooth = savgol_filter(D_raw, window_length=win, polyorder=smooth_poly)
    gradient = np.diff(D_smooth)
    is_dec = gradient < -gradient_threshold
    padded = np.concatenate(([False], is_dec, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0, D_raw, D_smooth, gradient
    longest = np.argmax(ends - starts)
    return int(starts[longest]), D_raw, D_smooth, gradient


def get_frame_at_sharpest_drop(
    actions: np.ndarray,
    smooth_window: int = 15,
    smooth_poly: int = 2,
    gradient_threshold: float = 1.0,
) -> tuple[None, int]:
    """
    Convenience wrapper around find_sharpest_drop that accepts a pre-loaded
    actions array and returns ``(None, frame_index)`` to match the calling
    convention used in build_trajectory_image.

    Args:
        actions:            (T, n_actions) float32 array for one episode.
        smooth_window:      Passed to find_sharpest_drop.
        smooth_poly:        Passed to find_sharpest_drop.
        gradient_threshold: Passed to find_sharpest_drop.

    Returns:
        (None, sharpest_drop_frame_index)
    """
    idx, _, _, _ = find_sharpest_drop(
        actions,
        smooth_window=smooth_window,
        smooth_poly=smooth_poly,
        gradient_threshold=gradient_threshold,
    )
    return None, idx
