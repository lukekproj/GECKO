"""Tests for data.data_calculations (GazeCalculator): spherical coords, angular velocity, FVR."""

import numpy as np
from data.data_calculations import GazeCalculator

# --- compute_spherical_coords ---

def test_spherical_coords_straight_down_gaze():
    rho, theta, phi = GazeCalculator.compute_spherical_coords(
        np.array([0.0]), np.array([0.0]), eye_height_m=0.2
    )
    np.testing.assert_allclose(rho, [0.2])
    np.testing.assert_allclose(phi, [0.0], atol=1e-10)

def test_spherical_coords_45_degree_gaze():
    rho, theta, phi = GazeCalculator.compute_spherical_coords(
        np.array([0.2]), np.array([0.0]), eye_height_m=0.2
    )
    np.testing.assert_allclose(rho, [0.2 * np.sqrt(2)])
    np.testing.assert_allclose(np.rad2deg(phi), [45.0])
    np.testing.assert_allclose(theta, [0.0], atol=1e-10)

def test_spherical_coords_rho_never_negative():
    x = np.array([-5.0, 0.0, 5.0, 100.0])
    y = np.array([3.0, -3.0, 0.0, -100.0])
    rho, _, _ = GazeCalculator.compute_spherical_coords(x, y, eye_height_m=0.2)
    assert np.all(rho >= 0)

# --- compute_fvr ---

def test_compute_fvr_matches_known_value():
    rho = np.array([1.0])
    epsilon = np.array([np.pi / 2])
    fvr = GazeCalculator.compute_fvr(rho, epsilon, visual_angle_deg=5.0)
    expected = 1.0 * np.tan(np.deg2rad(2.5))
    np.testing.assert_allclose(fvr, [expected])

def test_compute_fvr_scales_linearly_with_rho():
    epsilon = np.array([np.pi / 2])
    fvr_1 = GazeCalculator.compute_fvr(np.array([1.0]), epsilon, visual_angle_deg=5.0)
    fvr_2 = GazeCalculator.compute_fvr(np.array([2.0]), epsilon, visual_angle_deg=5.0)
    np.testing.assert_allclose(fvr_2, fvr_1 * 2.0)

# --- compute_epsilon_from_gaze_direction ---

def test_epsilon_zero_for_straight_down_gaze():
    eps = GazeCalculator.compute_epsilon_from_gaze_direction(
        np.array([0.0]), np.array([0.0]), eye_height_m=0.2
    )
    np.testing.assert_allclose(eps, [0.0], atol=1e-10)

def test_epsilon_increases_with_offset_gaze():
    eps_center = GazeCalculator.compute_epsilon_from_gaze_direction(
        np.array([0.0]), np.array([0.0]), eye_height_m=0.2
    )
    eps_offset = GazeCalculator.compute_epsilon_from_gaze_direction(
        np.array([0.5]), np.array([0.0]), eye_height_m=0.2
    )
    assert eps_offset[0] > eps_center[0]

# --- compute_angular_velocity ---

def test_angular_velocity_zero_for_stationary_gaze():
    n = 21  # must exceed sg_window default (11) for savgol_filter
    x = np.full(n, 0.1)
    y = np.full(n, 0.1)
    rho, theta, phi = GazeCalculator.compute_spherical_coords(x, y, eye_height_m=0.2)
    v_deg_s, phi_dot, theta_dot, rho_dot = GazeCalculator.compute_angular_velocity(
        x, y, rho, phi, frame_rate_hz=1000.0, eye_height_m=0.2
    )
    np.testing.assert_allclose(v_deg_s, np.zeros(n), atol=1e-6)

def test_angular_velocity_no_divide_by_zero_at_origin():
    n = 21
    x = np.linspace(-0.1, 0.1, n)
    y = np.zeros(n)
    rho, theta, phi = GazeCalculator.compute_spherical_coords(x, y, eye_height_m=0.2)
    v_deg_s, phi_dot, theta_dot, rho_dot = GazeCalculator.compute_angular_velocity(
        x, y, rho, phi, frame_rate_hz=1000.0, eye_height_m=0.2
    )
    assert not np.any(np.isnan(v_deg_s))
    assert not np.any(np.isinf(v_deg_s))
    assert not np.any(np.isnan(theta_dot))
    assert not np.any(np.isinf(theta_dot))

def test_angular_velocity_returns_nonnegative_magnitude():
    n = 21
    x = np.linspace(-0.2, 0.2, n)
    y = np.linspace(0.1, -0.1, n)
    rho, theta, phi = GazeCalculator.compute_spherical_coords(x, y, eye_height_m=0.2)
    v_deg_s, *_ = GazeCalculator.compute_angular_velocity(
        x, y, rho, phi, frame_rate_hz=1000.0, eye_height_m=0.2
    )
    assert np.all(v_deg_s >= 0)