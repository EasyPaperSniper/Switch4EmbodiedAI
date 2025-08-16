from collections import deque
import numpy as np

class QposStreamSmoother:
    """
    Streaming smoother for MuJoCo-style qpos with quaternion-aware filtering.

    qpos layout:
      - qpos[0:3]  : root position (xyz)
      - qpos[3:7]  : root orientation quaternion (4 elems)
      - qpos[7:]   : remaining DOF positions

    This uses a trailing window (causal) moving average for positions/DOFs,
    and a sign-consistent average for quaternions.
    """
    def __init__(self,  window_size: int = 7):
        self.dim = None
        self.window = int(window_size)
        self.buf_pos = deque(maxlen=self.window)     # (3,)
        self.buf_quat = deque(maxlen=self.window)    # (4,) aligned
        self.buf_dofs = deque(maxlen=self.window)    # (D-7,)

        # Running sums for O(1) averaging
        self.sum_pos = np.zeros(3, dtype=np.float64)
        self.sum_quat = np.zeros(4, dtype=np.float64)  # sum of aligned quats
        self.sum_dofs = None  # lazy-init when first qpos arrives

        self.last_aligned_quat = None  # (4,)
        self.length = 0  # current buffer length

    # ---------- utilities ----------
    @staticmethod
    def _normalize_quat(q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q) + 1e-12
        return q / n

    @staticmethod
    def _align_quat_to_ref(q, ref):
        """Flip sign of q if dot(q, ref) < 0 to keep hemisphere continuity."""
        if ref is None:
            return q
        return q if float(np.dot(q, ref)) >= 0.0 else -q

    # ---------- main API ----------
    def add(self, qpos: np.ndarray) -> np.ndarray:
        """
        Add a new qpos sample and return the current smoothed qpos.
        """
        if self.dim is None:
            self.dim = qpos.shape[0]
        qpos = np.asarray(qpos, dtype=np.float64)
        assert qpos.shape[-1] == self.dim, f"expected dim={self.dim}, got {qpos.shape[-1]}"

        pos = qpos[0:3]
        quat_raw = qpos[3:7]
        dofs = qpos[7:] if self.dim > 7 else np.empty((0,), dtype=np.float64)

        # Lazy init sums/dofs length
        if self.sum_dofs is None:
            self.sum_dofs = np.zeros_like(dofs, dtype=np.float64)

        # Normalize and align quaternion to last aligned for sign continuity
        quat = self._normalize_quat(quat_raw)
        quat_aligned = self._align_quat_to_ref(quat, self.last_aligned_quat)
        # Update last aligned reference
        self.last_aligned_quat = quat_aligned

        # If buffers are full, pop left and subtract from sums
        if self.length == self.window:
            old_pos = self.buf_pos.popleft()
            old_quat = self.buf_quat.popleft()
            old_dofs = self.buf_dofs.popleft()
            self.sum_pos -= old_pos
            self.sum_quat -= old_quat
            self.sum_dofs -= old_dofs
            # length remains the same (full)

        else:
            self.length += 1

        # Push new entries and update sums
        self.buf_pos.append(pos)
        self.buf_quat.append(quat_aligned)
        self.buf_dofs.append(dofs)

        self.sum_pos += pos
        self.sum_quat += quat_aligned
        self.sum_dofs += dofs

        # Compute averages
        L = float(self.length)
        pos_avg = self.sum_pos / L
        dofs_avg = self.sum_dofs / L
        quat_avg = self.sum_quat / L
        quat_avg = self._normalize_quat(quat_avg)

        # Compose smoothed qpos
        if dofs_avg.size > 0:
            smoothed = np.concatenate([pos_avg, quat_avg, dofs_avg], axis=0)
        else:
            smoothed = np.concatenate([pos_avg, quat_avg], axis=0)

        return smoothed.astype(qpos.dtype, copy=False)

    def current(self) -> np.ndarray:
        """
        Return the current smoothed qpos without adding a new sample.
        If buffer is empty, raises ValueError.
        """
        if self.length == 0:
            raise ValueError("No samples in buffer")
        L = float(self.length)
        pos_avg = self.sum_pos / L
        dofs_avg = (self.sum_dofs / L) if (self.sum_dofs is not None and self.sum_dofs.size) else np.empty((0,))
        quat_avg = self._normalize_quat(self.sum_quat / L)
        return np.concatenate([pos_avg, quat_avg, dofs_avg], axis=0)

    def reset(self):
        """Clear internal state."""
        self.buf_pos.clear()
        self.buf_quat.clear()
        self.buf_dofs.clear()
        self.sum_pos[:] = 0.0
        self.sum_quat[:] = 0.0
        self.sum_dofs = None
        self.last_aligned_quat = None
        self.length = 0