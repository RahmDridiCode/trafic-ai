from collections import defaultdict, deque
import math
from statistics import median


class _TrackState:
    __slots__ = (
        "accepted",
        "last_speed",
        "last_fresh_frame",
        "fresh",
        "consecutive_position_rejections",
        "provisional_speed_value",
        "status",
    )

    def __init__(self, maxlen):
        self.accepted = deque(maxlen=maxlen)  # (frame, world_pt, pixel_pt, box_width)
        self.last_speed = None
        self.last_fresh_frame = None
        self.fresh = False
        self.consecutive_position_rejections = 0
        self.provisional_speed_value = None
        self.status = "na"


class SpeedEstimator:
    """Computes displacement per second.

    Units: metres/s when a homography transformer is set, pixels/s otherwise.
    No km/h conversion happens here; convert exactly once at display/export time
    (see visualizer.speed_to_display).

    Time comes from real frame indices, so frame skipping is handled correctly.

    update() returns confirmed speeds, short-lived held speeds, or None. Provisional speeds
    are available separately through provisional_speed(track_id):
      - provisional: `provisional_warmup` accepted observations and a window of at least
        `provisional_window_seconds`;
      - confirmed: `warmup` accepted observations and a window of at least `min_window_seconds`;
      - confirmed speeds take precedence; held speeds never count as fresh.

    Robustness against unstable detector boxes:
      - position outliers: predict the next bottom-centre from median recent per-frame velocities;
        reject only when the observation is more than 0.3 median box widths from that prediction;
        after three consecutive rejections, accept the observation and restart the window;
      - speed = displacement between the per-axis MEDIAN positions of the oldest and newest third
        of the window;
      - dead zone: the pixel displacement must exceed max(`dead_zone_sigma` x noise of the
        bottom-centre positions, `min_move_ratio` x vehicle width), otherwise the speed is 0.
        Noise = 1.4826 x MAD of residuals after removing a linear trend (real motion is not noise);
        the width floor covers correlated box drift, which the noise estimate cannot see;
      - a gap longer than `max_gap_seconds` restarts the warm-up of that track.
    The window holds `smoothing` observations (more at high FPS so it can span `min_window_seconds`).
    """

    def __init__(self, fps, transformer=None, smoothing=10, max_speed=None, min_speed=None,
                 warmup=6, prediction_tolerance_ratio=0.3, dead_zone_sigma=2.5, min_move_ratio=0.1,
                 min_window_seconds=0.4, max_gap_seconds=1.0, frame_step=1,
                 edge_margin=3, stale_seconds=0.5, provisional_warmup=4,
                 provisional_window_seconds=0.2, edge_hold_seconds=1.0,
                 reject_hold_seconds=0.5):
        self.fps = fps
        self.transformer = transformer
        self.warmup = max(2, int(warmup))
        self.min_window_seconds = min_window_seconds
        self.provisional_warmup = max(2, int(provisional_warmup))
        self.provisional_window_seconds = max(0.0, float(provisional_window_seconds))
        window = max(
            int(smoothing),
            self.warmup,
            math.ceil(min_window_seconds * fps) + 2,
            self.provisional_warmup,
            math.ceil(self.provisional_window_seconds * fps) + 2,
        )
        self.max_speed = max_speed
        self.min_speed = min_speed if min_speed is not None else (0.5 if transformer else 0.0)
        self.prediction_tolerance_ratio = prediction_tolerance_ratio
        self.dead_zone_sigma = dead_zone_sigma
        self.min_move_ratio = min_move_ratio
        self.frame_step = max(1, int(frame_step))
        self.max_gap_frames = max(1.0, max_gap_seconds * fps)
        self.max_gap_frames = min(self.max_gap_frames, 3 * self.frame_step)
        self.edge_margin = max(0, int(edge_margin))
        self.edge_hold_frames = max(0.0, float(edge_hold_seconds) * fps)
        # Keep the older stale_seconds keyword as an alias for rejection holds.
        reject_hold = (
            stale_seconds
            if stale_seconds != 0.5 and reject_hold_seconds == 0.5
            else reject_hold_seconds
        )
        self.reject_hold_frames = max(0.0, float(reject_hold) * fps)
        self.states = defaultdict(lambda: _TrackState(window))
        self._auto_frame = 0

    def is_fresh(self, track_id):
        """Return whether the latest update produced a fresh confirmed speed."""
        state = self.states.get(track_id)
        return bool(state and state.fresh)

    def provisional_speed(self, track_id):
        """Return the latest provisional speed, unless a confirmed speed took over."""
        state = self.states.get(track_id)
        if state is None or state.status != "provisional":
            return None
        return state.provisional_speed_value

    def speed_status(self, track_id):
        """Return the current display status: confirmed, provisional, held, or na."""
        state = self.states.get(track_id)
        return state.status if state is not None else "na"

    def _held_speed(self, state, frame_no, hold_frames):
        if state.last_fresh_frame is None:
            return None
        if state.last_speed is None or frame_no - state.last_fresh_frame > hold_frames:
            return None
        state.status = "held"
        return state.last_speed

    def _touches_speed_edge(self, box, image_size):
        if box is None or image_size is None:
            return False
        x, y, width, height = box
        image_width, image_height = image_size
        margin = self.edge_margin
        return (
            x <= margin
            or x + width >= image_width - margin
            or y + height >= image_height - margin
        )

    @staticmethod
    def _robust_noise(frames, values):
        """1.4826 x MAD of the residuals of a least-squares line values ~ frames."""
        n = len(frames)
        mf = sum(frames) / n
        mv = sum(values) / n
        sff = sum((f - mf) ** 2 for f in frames)
        slope = sum((f - mf) * (v - mv) for f, v in zip(frames, values)) / sff if sff else 0.0
        res = [v - (mv + slope * (f - mf)) for f, v in zip(frames, values)]
        med = median(res)
        return 1.4826 * median(abs(r - med) for r in res)

    def _is_position_outlier(self, st, frame_no, pixel_pt, box_width):
        if len(st.accepted) < 3 or not box_width:
            return False
        recent = [o for o in list(st.accepted)[-5:] if o[3]]
        if len(recent) < 3:
            return False
        velocities = []
        for previous, current in zip(recent, recent[1:]):
            elapsed_frames = current[0] - previous[0]
            if elapsed_frames <= 0:
                continue
            velocities.append((
                (current[2][0] - previous[2][0]) / elapsed_frames,
                (current[2][1] - previous[2][1]) / elapsed_frames,
            ))
        if not velocities:
            return False

        last = recent[-1]
        elapsed_frames = frame_no - last[0]
        predicted = (
            last[2][0] + median(v[0] for v in velocities) * elapsed_frames,
            last[2][1] + median(v[1] for v in velocities) * elapsed_frames,
        )
        median_width = median(o[3] for o in recent)
        deviation = math.hypot(pixel_pt[0] - predicted[0], pixel_pt[1] - predicted[1])
        return deviation > self.prediction_tolerance_ratio * median_width

    def update(self, track_id, image_point, frame_no=None, ref_size=None, box_size=None,
               box=None, img_size=None):
        """Update one track from its ground point and optional image bounds.

        frame_no is the original video frame index. box_size is (w, h) for position
        filtering; box is (x, y, w, h), and img_size is (image width, image height).
        """
        if frame_no is None:
            frame_no = self._auto_frame
            self._auto_frame += 1
        st = self.states[track_id]
        st.fresh = False
        st.status = "na"
        st.provisional_speed_value = None
        if st.accepted and frame_no <= st.accepted[-1][0]:
            return self._held_speed(st, frame_no, self.reject_hold_frames)
        if self._touches_speed_edge(box, img_size):
            st.consecutive_position_rejections = 0
            return self._held_speed(st, frame_no, self.edge_hold_frames)

        if st.accepted and frame_no - st.accepted[-1][0] > self.max_gap_frames:
            st.accepted.clear()
            st.consecutive_position_rejections = 0

        box_width = box_size[0] if box_size is not None else None
        if ref_size is None:
            ref_size = box_width

        if st.accepted and box_width and st.accepted[-1][3]:
            previous_width = st.accepted[-1][3]
            if abs(box_width - previous_width) / previous_width > 0.25:
                st.accepted.clear()
                st.consecutive_position_rejections = 0

        if self._is_position_outlier(st, frame_no, image_point, box_width):
            st.consecutive_position_rejections += 1
            if st.consecutive_position_rejections < 3:
                return self._held_speed(st, frame_no, self.reject_hold_frames)
            st.accepted.clear()
        st.consecutive_position_rejections = 0
        world = self.transformer.transform_point(image_point) if self.transformer else image_point
        st.accepted.append((frame_no, world, image_point, box_width))

        history = list(st.accepted)
        span = (history[-1][0] - history[0][0]) / self.fps
        if len(history) >= self.warmup and span >= self.min_window_seconds:
            speed = self._speed(history, ref_size)
            if self.max_speed is not None and speed > self.max_speed:
                return self._held_speed(st, frame_no, self.reject_hold_frames)
            st.last_speed = speed
            st.last_fresh_frame = frame_no
            st.fresh = True
            st.status = "confirmed"
            return speed

        if (
            len(history) >= self.provisional_warmup
            and span >= self.provisional_window_seconds
        ):
            speed = self._speed(history, ref_size)
            if self.max_speed is None or speed <= self.max_speed:
                st.provisional_speed_value = speed
                st.status = "provisional"
            else:
                return self._held_speed(st, frame_no, self.reject_hold_frames)
        return None

    def _speed(self, history, ref_size=None):
        k = max(1, len(history) // 3)
        head, tail = history[:k], history[-k:]
        dt = (median(o[0] for o in tail) - median(o[0] for o in head)) / self.fps
        if dt <= 0:
            return 0.0

        def med_pt(items, idx):
            return (median(o[idx][0] for o in items), median(o[idx][1] for o in items))

        pa, pb = med_pt(head, 2), med_pt(tail, 2)
        frames = [o[0] for o in history]
        noise = math.hypot(
            self._robust_noise(frames, [o[2][0] for o in history]),
            self._robust_noise(frames, [o[2][1] for o in history]),
        )
        threshold = max(self.dead_zone_sigma * noise, self.min_move_ratio * (ref_size or 0.0))
        if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) < threshold:
            return 0.0
        a, b = med_pt(head, 1), med_pt(tail, 1)
        speed = math.hypot(b[0] - a[0], b[1] - a[1]) / dt
        return 0.0 if speed < self.min_speed else speed
