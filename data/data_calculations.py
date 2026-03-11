"""
Data Calculations Module

Computes spherical gaze metrics, angular velocity, and foveal visual radius (FVR)
from KINARM gaze point-of-regard channels (Gaze_X, Gaze_Y).

Coordinate/units assumptions:
- Gaze_X and Gaze_Y are point-of-regard coordinates on the stimulus plane (meters).
- The stimulus plane is at z = 0.
- The eye is at a fixed height H above the plane (meters).
- We use an eye-centered coordinate frame where:
    x' = x
    y' = y
    z' = H
  (i.e., the gaze point is on the plane and the eye is offset by height H.)

Angles:
- theta (azimuth) and phi (elevation) are computed in radians.
- Angular speed magnitude is returned in deg/s.

This module also includes GUI helper functions that compute and plot metrics
for the currently selected trial in KinarmDataExplorer.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# All values imported from user_prefs.py for lab-level configurability.
from utility.user_prefs import (
    DEFAULT_EYE_HEIGHT_M,
    DEFAULT_VISUAL_ANGLE_DEG,
    DEFAULT_SAVGOL_WINDOW,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_GAZE_LOWPASS_CUTOFF_HZ,
    DEFAULT_GAZE_LOWPASS_ORDER
)

class GazeCalculator:
    """
    Gaze-related coordinate transforms and metric calculations.

    All functions are stateless and operate on numpy arrays.
    """

    @staticmethod
    def compute_spherical_coords(
        x: np.ndarray,
        y: np.ndarray,
        eye_height_m: float = DEFAULT_EYE_HEIGHT_M,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert eye-centered Cartesian (x', y', z') to spherical (rho, theta, phi).

        Eq. (2):  eye-centered transform: x' = x, y' = y, z' = H
        Eq. (3a): rho
        Eq. (3b): theta = atan2(y', x')   (radians)
        Eq. (3c): phi   = acos(z' / rho)  (radians)

        Parameters
        ----------
        x : np.ndarray
            Gaze_X on stimulus plane (meters).
        y : np.ndarray
            Gaze_Y on stimulus plane (meters).
        eye_height_m : float
            Eye height above stimulus plane H (meters).

        Returns
        -------
        rho_m : np.ndarray
            Radial distance (meters).
        theta_rad : np.ndarray
            Azimuth angle (radians).
        phi_rad : np.ndarray
            Elevation angle (radians).
        """
        # Eq. (2): eye-centered Cartesian coordinates (x′, y′, z′)
        x_eye = np.asarray(x, dtype=float)
        y_eye = np.asarray(y, dtype=float)
        z_eye = np.full_like(x_eye, eye_height_m, dtype=float)

        # Eq. (3a): radial distance rho
        rho = np.sqrt(x_eye**2 + y_eye**2 + z_eye**2)

        # Eq. (3b): azimuth theta (radians)
        theta_rad = np.arctan2(y_eye, x_eye)

        # Eq. (3c): elevation phi (radians)
        # Small numerical drift can push (z/rho) slightly outside [-1,1].
        phi_rad = np.arccos(np.clip(z_eye / rho, -1.0, 1.0))

        return rho, theta_rad, phi_rad

    @staticmethod
    def compute_angular_velocity(
        x: np.ndarray,
        y: np.ndarray,
        rho: np.ndarray,
        phi_rad: np.ndarray,
        frame_rate_hz: float,
        eye_height_m: float = DEFAULT_EYE_HEIGHT_M,
        sg_window: int = DEFAULT_SAVGOL_WINDOW,
        sg_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute angular speed magnitude using Eq. (6a).

        Uses Eq. (4a) and Eq. (4b) to compute phi_dot and theta_dot in rad/s,
        then combines them into angular speed magnitude (Eq. 6a) in deg/s.

        Parameters
        ----------
        x : np.ndarray
            Gaze_X on stimulus plane (meters).
        y : np.ndarray
            Gaze_Y on stimulus plane (meters).
        rho : np.ndarray
            Radial distance (meters) from ``compute_spherical_coords``.
        phi_rad : np.ndarray
            Elevation angle (radians) from ``compute_spherical_coords``.
        frame_rate_hz : float
            Sampling rate in Hz.
        eye_height_m : float
            Eye height H in meters.
        sg_window : int
            Savitzky-Golay window length (must be odd).
        sg_polyorder : int
            Savitzky-Golay polynomial order.

        Returns
        -------
        v_deg_s : np.ndarray
            Angular speed magnitude (deg/s).
        phi_dot_rad_s : np.ndarray
            Elevation rate (rad/s).
        theta_dot_rad_s : np.ndarray
            Azimuth rate (rad/s).
        rho_dot_m_s : np.ndarray
            Radial rate (m/s).
        """
        x_eye = np.asarray(x, dtype=float)
        y_eye = np.asarray(y, dtype=float)
        z_eye = np.full_like(x_eye, eye_height_m, dtype=float)

        dt = 1.0 / float(frame_rate_hz)

        # Time derivatives of eye-centered Cartesian coordinates (Savitzky-Golay)
        x_dot = savgol_filter(x_eye, sg_window, sg_polyorder, deriv=1, delta=dt)
        y_dot = savgol_filter(y_eye, sg_window, sg_polyorder, deriv=1, delta=dt)
        z_dot = np.zeros_like(x_eye)  # eye height is constant; derivative is zero
        rho_dot = savgol_filter(np.asarray(rho, dtype=float), sg_window, sg_polyorder, deriv=1, delta=dt)

        # Common terms
        denom_xy = (x_eye**2 + y_eye**2)
        denom_xyz = (x_eye**2 + y_eye**2 + z_eye**2)
        sqrt_xy = np.sqrt(denom_xy)

        # Eq. (4a): elevation angle rate phi_dot (rad/s)
        numerator = (z_eye * (x_eye * x_dot + y_eye * y_dot) - denom_xy * z_dot)
        denominator = denom_xyz * sqrt_xy
        phi_dot = numerator / denominator

        # Eq. (4b): azimuth angle rate theta_dot (rad/s)
        theta_dot = (x_dot * y_eye - x_eye * y_dot) / denom_xy

        # Eq. (6a): angular speed magnitude (rad/s), then convert to deg/s
        v_rad_s = np.sqrt((theta_dot * np.sin(phi_rad))**2 + (phi_dot)**2)
        v_deg_s = np.rad2deg(v_rad_s)

        return v_deg_s, phi_dot, theta_dot, rho_dot

    @staticmethod
    def compute_fvr(
        rho_m: np.ndarray,
        epsilon_rad: np.ndarray,
        visual_angle_deg: float = DEFAULT_VISUAL_ANGLE_DEG,
    ) -> np.ndarray:
        """
        Compute foveal visual radius (FVR).

        Uses the standard relationship between viewing distance, cone angle,
        and surface slant (epsilon).

        Parameters
        ----------
        rho_m : np.ndarray
            Eye-to-gaze distance (meters).
        epsilon_rad : np.ndarray
            Angle between gaze direction and stimulus normal (radians).
        visual_angle_deg : float
            Foveal cone angle (degrees).

        Returns
        -------
        fvr_m : np.ndarray
            Foveal visual radius (meters).
        """
        delta_rad = np.deg2rad(visual_angle_deg)
        fvr = rho_m * np.tan(delta_rad / 2.0) / np.sin(epsilon_rad)
        return fvr

    @staticmethod
    def compute_epsilon_from_gaze_direction(
        x: np.ndarray,
        y: np.ndarray,
        eye_height_m: float = DEFAULT_EYE_HEIGHT_M,
        stimulus_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> np.ndarray:
        """
        Compute epsilon: angle between gaze direction and stimulus plane normal.

        Forms the eye-to-point vector in eye-centered coordinates:
            v = [x', y', z'] = [x, y, H]
        then computes the angle to the stimulus normal.

        Parameters
        ----------
        x : np.ndarray
            Gaze_X on stimulus plane (meters).
        y : np.ndarray
            Gaze_Y on stimulus plane (meters).
        eye_height_m : float
            Eye height H (meters).
        stimulus_normal : np.ndarray
            Normal vector of the stimulus plane.

        Returns
        -------
        epsilon_rad : np.ndarray
            Angle to surface normal (radians).
        """
        x_eye = np.asarray(x, dtype=float)
        y_eye = np.asarray(y, dtype=float)
        z_eye = np.full_like(x_eye, eye_height_m, dtype=float)

        gaze_vecs = np.stack([x_eye, y_eye, z_eye], axis=1)
        gaze_norms = np.linalg.norm(gaze_vecs, axis=1, keepdims=True)
        gaze_dirs = gaze_vecs / gaze_norms

        stim_normal = np.asarray(stimulus_normal, dtype=float)
        stim_normal = stim_normal / np.linalg.norm(stim_normal)

        dot_products = gaze_dirs @ stim_normal
        epsilon_rad = np.arccos(np.clip(dot_products, -1.0, 1.0))
        return epsilon_rad

def calculate_gaze_metrics(explorer) -> None:
    """
    Compute and plot rho/theta/phi for the selected trial.

    Uses interpolated Gaze_X/Gaze_Y, applies a light low-pass filter, then
    computes spherical coordinates in radians (rho in meters).
    """
    if not explorer.current_trial:
        print("No trial selected!")
        return

    try:
        frame_rate = float(explorer.current_trial.frame_rate)

        interpolated_data = explorer.get_interpolated_gaze_data()
        if interpolated_data is None:
            return

        gx = np.asarray(interpolated_data["Gaze_X"], dtype=float)
        gy = np.asarray(interpolated_data["Gaze_Y"], dtype=float)

        # Optional smoothing before derivatives / angles
        gx = explorer.lowpass_filter(gx, cutoff=DEFAULT_GAZE_LOWPASS_CUTOFF_HZ, fs=frame_rate, order=DEFAULT_GAZE_LOWPASS_ORDER)
        gy = explorer.lowpass_filter(gy, cutoff=DEFAULT_GAZE_LOWPASS_CUTOFF_HZ, fs=frame_rate, order=DEFAULT_GAZE_LOWPASS_ORDER)

        rho, theta, phi = explorer.gaze_calculator.compute_spherical_coords(gx, gy)

        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(rho, label="rho (m)")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(theta, label="theta (rad)", color='orange')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(phi, label="phi (rad)", color='green')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show(block=False)

    except Exception as e:
        print(f"Error computing gaze metrics: {e}")


def calculate_angular_velocity(explorer) -> None:
    """
    Compute and plot gaze angular velocity magnitude for the selected trial.

    Expected ranges (prof reminder):
    - Smooth pursuit: ~10–100 deg/s
    - Saccades: ~100–1200 deg/s
    """
    if not explorer.current_trial:
        print("No trial selected!")
        return

    try:
        frame_rate = float(explorer.current_trial.frame_rate)

        interpolated_data = explorer.get_interpolated_gaze_data()
        if interpolated_data is None:
            return

        gx = np.asarray(interpolated_data["Gaze_X"], dtype=float)
        gy = np.asarray(interpolated_data["Gaze_Y"], dtype=float)

        gx = explorer.lowpass_filter(gx, cutoff=DEFAULT_GAZE_LOWPASS_CUTOFF_HZ, fs=frame_rate, order=DEFAULT_GAZE_LOWPASS_ORDER)
        gy = explorer.lowpass_filter(gy, cutoff=DEFAULT_GAZE_LOWPASS_CUTOFF_HZ, fs=frame_rate, order=DEFAULT_GAZE_LOWPASS_ORDER)

        rho, theta, phi = explorer.gaze_calculator.compute_spherical_coords(gx, gy)
        v_deg_s, phi_dot, theta_dot, rho_dot = explorer.gaze_calculator.compute_angular_velocity(
            gx, gy, rho, phi, frame_rate_hz=frame_rate
        )

        plt.figure(figsize=(12, 6))
        plt.plot(v_deg_s, label="Angular Velocity (deg/s)")
        plt.title("Angular Velocity")
        plt.xlabel("Frame")
        plt.ylabel("deg/s")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    except Exception as e:
        print(f"Error computing angular velocity: {e}")

def calculate_fvr(explorer) -> None:
    """
    Compute and plot foveal visual radius (FVR) for the selected trial.

    NOTE: With Gaze_X/Y and eye height in meters, FVR is returned in meters.
    """
    if not explorer.current_trial:
        print("No trial selected!")
        return

    try:
        frame_rate = float(explorer.current_trial.frame_rate)

        interpolated_data = explorer.get_interpolated_gaze_data()
        if interpolated_data is None:
            return

        gx = np.asarray(interpolated_data["Gaze_X"], dtype=float)
        gy = np.asarray(interpolated_data["Gaze_Y"], dtype=float)

        # Optional smoothing (helps if epsilon gets noisy)
        gx = explorer.lowpass_filter(gx, cutoff=DEFAULT_GAZE_LOWPASS_CUTOFF_HZ, fs=frame_rate, order=DEFAULT_GAZE_LOWPASS_ORDER)
        gy = explorer.lowpass_filter(gy, cutoff=DEFAULT_GAZE_LOWPASS_CUTOFF_HZ, fs=frame_rate, order=DEFAULT_GAZE_LOWPASS_ORDER)

        rho, _, _ = explorer.gaze_calculator.compute_spherical_coords(gx, gy)
        epsilon_rad = explorer.gaze_calculator.compute_epsilon_from_gaze_direction(gx, gy)
        fvr_m = explorer.gaze_calculator.compute_fvr(rho, epsilon_rad)

        plt.figure(figsize=(12, 6))
        plt.plot(fvr_m, label="FVR (m)")
        plt.title("Foveal Visual Radius (FVR)")
        plt.xlabel("Frame")
        plt.ylabel("meters")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    except Exception as e:
        print(f"Error computing FVR: {e}")