"""
utils/smoothing.py
===================
Signal-smoothing primitives used to turn noisy, jittery landmark coordinates
into smooth, natural cursor motion.

Two smoothers are provided:

* ``ExponentialSmoother`` — simple, cheap, one weight to tune. Good default.
* ``OneEuroFilter``       — adapts to speed: heavy smoothing when the hand is
                             nearly still (kills jitter), light smoothing when
                             moving fast (kills lag). Used for the cursor by
                             default because it gives the most "natural" feel.

Both operate on plain floats so they can smooth x and y independently (or be
combined into a Point2D smoother below).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


class ExponentialSmoother:
    """Simple exponential moving average smoother."""

    def __init__(self, factor: float = 0.5):
        self.factor = max(0.01, min(1.0, factor))
        self._value: float | None = None

    def update(self, new_value: float) -> float:
        if self._value is None:
            self._value = new_value
        else:
            self._value = self.factor * new_value + (1 - self.factor) * self._value
        return self._value

    def reset(self, value: float | None = None) -> None:
        self._value = value


def _low_pass(prev: float | None, value: float, alpha: float) -> float:
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev


class OneEuroFilter:
    """
    A speed-adaptive low-pass filter (Casiez et al., 2012).

    Great for cursor tracking: minimizes jitter on a still hand while
    staying responsive during fast intentional motion.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.4, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(self, value: float, timestamp: float | None = None) -> float:
        t = timestamp if timestamp is not None else time.time()
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = value
            self._dx_prev = 0.0
            return value

        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        dx = (value - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = _low_pass(self._dx_prev, dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = _low_pass(self._x_prev, value, a)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


@dataclass
class Point2D:
    x: float
    y: float


class PointSmoother:
    """Smooths a 2D point using two independent One-Euro filters."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.4, d_cutoff: float = 1.0):
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def update(self, x: float, y: float) -> Point2D:
        t = time.time()
        return Point2D(self._fx.update(x, t), self._fy.update(y, t))

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
