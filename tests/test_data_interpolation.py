"""Tests for data.data_interpolation: sentinel cleaning, gap detection, linear interpolation."""

import numpy as np
import pytest

from data.data_interpolation import (
    _sanitize_kinarm_signal,
    _should_sanitize_channel,
    _find_nan_gaps,
    _linear_interpolate_gap,
    _expand_nan_regions,
)

# --- _sanitize_kinarm_signal ---

def test_sanitize_replaces_sentinel_values_with_nan():
    data = np.array([1.0, 2.0, 99.9, 3.0, -99.9])
    result = _sanitize_kinarm_signal(data)
    assert np.isnan(result[2])
    assert np.isnan(result[4])
    assert result[0] == 1.0
    assert result[1] == 2.0
    assert result[3] == 3.0

def test_sanitize_leaves_valid_values_untouched():
    data = np.array([0.0, 50.0, -50.0, 99.8, -99.8])
    result = _sanitize_kinarm_signal(data)
    np.testing.assert_array_equal(result, data)

def test_sanitize_does_not_mutate_input():
    data = np.array([1.0, 99.9, 3.0])
    original = data.copy()
    _sanitize_kinarm_signal(data)
    np.testing.assert_array_equal(data, original)

# --- _should_sanitize_channel ---

@pytest.mark.parametrize("channel_name", ["Gaze_X", "Gaze_Y", "gaze_x"])
def test_should_sanitize_gaze_position_channels(channel_name):
    assert _should_sanitize_channel(channel_name) is True

def test_should_not_sanitize_gaze_timestamp():
    assert _should_sanitize_channel("Gaze_TimeStamp") is False

@pytest.mark.parametrize("channel_name", [
    "xT", "yT", "Right_Shoulder", "Elbow_Velocity", "TP_TABLE",
])
def test_should_not_sanitize_non_gaze_channels(channel_name):
    assert _should_sanitize_channel(channel_name) is False

# --- _find_nan_gaps ---

def test_find_nan_gaps_no_gaps():
    data = np.array([1.0, 2.0, 3.0])
    assert _find_nan_gaps(data) == []

def test_find_nan_gaps_single_gap():
    data = np.array([1.0, np.nan, np.nan, 4.0])
    gaps = _find_nan_gaps(data)
    assert len(gaps) == 1
    assert gaps[0].start == 1
    assert gaps[0].end == 2
    assert gaps[0].length == 2

def test_find_nan_gaps_multiple_gaps():
    data = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    gaps = _find_nan_gaps(data)
    assert len(gaps) == 2
    assert gaps[0].length == 1
    assert gaps[1].length == 2

# --- _linear_interpolate_gap ---

def test_linear_interpolate_gap_interior():
    data = np.array([0.0, np.nan, np.nan, 6.0])
    gap_indices = np.array([1, 2])
    result = _linear_interpolate_gap(data, gap_indices)
    np.testing.assert_allclose(result[1], 2.0)
    np.testing.assert_allclose(result[2], 4.0)

def test_linear_interpolate_gap_at_start_uses_flat_extrapolation():
    data = np.array([np.nan, np.nan, 5.0, 5.0])
    gap_indices = np.array([0, 1])
    result = _linear_interpolate_gap(data, gap_indices)
    assert result[0] == 5.0
    assert result[1] == 5.0

def test_linear_interpolate_gap_at_end_uses_flat_extrapolation():
    data = np.array([3.0, 3.0, np.nan, np.nan])
    gap_indices = np.array([2, 3])
    result = _linear_interpolate_gap(data, gap_indices)
    assert result[2] == 3.0
    assert result[3] == 3.0

# --- _expand_nan_regions ---

def test_expand_nan_regions_buffers_both_sides():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _expand_nan_regions(data, buffer_frames=2)
    expected_nan_indices = {3, 4, 5, 6, 7}
    actual_nan_indices = set(np.where(np.isnan(result))[0])
    assert actual_nan_indices == expected_nan_indices

def test_expand_nan_regions_zero_buffer_is_noop():
    data = np.array([1.0, np.nan, 3.0])
    result = _expand_nan_regions(data, buffer_frames=0)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(data))

def test_expand_nan_regions_no_nan_returns_unchanged():
    data = np.array([1.0, 2.0, 3.0])
    result = _expand_nan_regions(data, buffer_frames=5)
    np.testing.assert_array_equal(result, data)

def test_expand_nan_regions_handles_gap_at_array_start():
    data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
    result = _expand_nan_regions(data, buffer_frames=5)
    assert np.all(np.isnan(result))

def test_expand_nan_regions_handles_gap_at_array_end():
    data = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
    result = _expand_nan_regions(data, buffer_frames=5)
    assert np.all(np.isnan(result))

def test_expand_nan_regions_merges_nearby_gaps():
    data = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    result = _expand_nan_regions(data, buffer_frames=2)
    assert np.all(np.isnan(result))

def test_expand_nan_regions_does_not_mutate_input():
    data = np.array([1.0, np.nan, 3.0])
    original_nan_mask = np.isnan(data).copy()
    _expand_nan_regions(data, buffer_frames=1)
    np.testing.assert_array_equal(np.isnan(data), original_nan_mask)