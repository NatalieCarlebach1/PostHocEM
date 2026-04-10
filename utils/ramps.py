"""Consistency ramp-up schedules (standard in SSL literature, e.g. mean-teacher)."""

import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Gradually ramp from 0 → 1 over rampup_length steps using a sigmoid."""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    return float(np.clip(current / rampup_length, 0.0, 1.0))


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
