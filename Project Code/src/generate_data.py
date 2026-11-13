"""
generate_data.py
Generates realistic multi-satellite telemetry data with nominal orbital dynamics
and controlled injected anomalies for benchmark demonstrations and model evaluation.
"""

import os
import random
import numpy as np
import pandas as pd


def generate_synthetic_telemetry(
    num_records_per_satellite: int = 500,
    satellites: list = None,
    anomaly_ratio: float = 0.05,
    random_state: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate realistic multi-satellite telemetry with orbital periodicity,
    sensor noise, and controlled anomalies with ground-truth labels.

    Features:
      - temperature (°C): Nominal ~66-74°C with diurnal eclipse cycle
      - voltage (V): Nominal 3.70-4.00V battery charge/discharge cycle
      - current (A): Nominal 1.65-2.05A solar array / payload load
      - orientation (°): Nominal -20° to +20° reaction wheel attitude angle
    """
    if satellites is None:
        satellites = ["SAT-01", "SAT-02", "SAT-03"]

    np.random.seed(random_state)
    random.seed(random_state)

    all_dfs = []
    base_start = pd.Timestamp("2026-03-01 00:00:00")

    for sat_idx, sat in enumerate(satellites):
        # Base time grid: 1 minute sampling
        t = np.arange(num_records_per_satellite)
        time_index = pd.date_range(
            start=base_start + pd.Timedelta(hours=sat_idx * 6),
            periods=num_records_per_satellite,
            freq="1min"
        )

        # 90-minute orbital period simulation
        orbit_phase = 2.0 * np.pi * t / 90.0
        attitude_phase = 2.0 * np.pi * t / 45.0

        # Nominal telemetry baselines + Gaussian noise
        temperature = 70.0 + 4.0 * np.sin(orbit_phase) + np.random.normal(0, 0.45, num_records_per_satellite)
        voltage = 3.85 + 0.15 * np.cos(orbit_phase) + np.random.normal(0, 0.025, num_records_per_satellite)
        current = 1.85 + 0.20 * np.sin(orbit_phase) + np.random.normal(0, 0.025, num_records_per_satellite)
        orientation = 20.0 * np.sin(attitude_phase) + np.random.normal(0, 1.0, num_records_per_satellite)

        # Ground-truth tracking
        ground_truth = np.zeros(num_records_per_satellite, dtype=int)
        anomaly_type = ["Normal"] * num_records_per_satellite

        # Calculate number of anomalies per satellite
        n_anomalies = max(5, int(num_records_per_satellite * anomaly_ratio))
        per_type = max(1, n_anomalies // 5)

        # Inject controlled anomalies across 5 distinct categories:
        # 1. Thermal Spike: radiator malfunction (temp > 80°C)
        indices_thermal = np.linspace(25, num_records_per_satellite - 30, per_type, dtype=int)
        for idx in indices_thermal:
            temperature[idx] = np.random.uniform(85.0, 96.0)
            ground_truth[idx] = 1
            anomaly_type[idx] = "Thermal Spike"

        # 2. Voltage Collapse: battery cell degradation (voltage < 3.0V)
        indices_voltage = (indices_thermal + 15) % num_records_per_satellite
        for idx in indices_voltage:
            voltage[idx] = np.random.uniform(2.35, 2.82)
            ground_truth[idx] = 1
            anomaly_type[idx] = "Voltage Drop"

        # 3. Current Surge: payload short-circuit / flare (current > 2.5A)
        indices_current = (indices_thermal + 32) % num_records_per_satellite
        for idx in indices_current:
            current[idx] = np.random.uniform(2.75, 3.45)
            ground_truth[idx] = 1
            anomaly_type[idx] = "Current Surge"

        # 4. Attitude Anomaly: thruster tumble (|orientation| > 45°)
        indices_attitude = (indices_thermal + 48) % num_records_per_satellite
        for idx in indices_attitude:
            sign = np.random.choice([-1, 1])
            orientation[idx] = sign * np.random.uniform(52.0, 78.0)
            ground_truth[idx] = 1
            anomaly_type[idx] = "Attitude Anomaly"

        # 5. Multivariate Correlation Drift: individual values within safe bounds,
        # but joint combination is physically anomalous (ML detector targets this!)
        indices_multi = (indices_thermal + 65) % num_records_per_satellite
        for idx in indices_multi:
            temperature[idx] = 78.5   # Below 80°C threshold
            voltage[idx] = 3.12       # Above 3.0V threshold
            current[idx] = 2.38       # Below 2.5A threshold
            orientation[idx] = 38.0   # Within ±45° threshold
            ground_truth[idx] = 1
            anomaly_type[idx] = "Multivariate Drift"

        df_sat = pd.DataFrame({
            "timestamp": time_index,
            "satellite_id": sat,
            "temperature": np.round(temperature, 2),
            "voltage": np.round(voltage, 2),
            "current": np.round(current, 2),
            "orientation": np.round(orientation, 2),
            "ground_truth": ground_truth,
            "anomaly_type": anomaly_type,
            "data_source": "Synthetic"
        })
        all_dfs.append(df_sat)

    df_total = pd.concat(all_dfs, ignore_index=True)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df_total.to_csv(output_path, index=False)
        print(f"✅ Generated {len(df_total)} synthetic telemetry records saved to: {output_path}")

    return df_total


def generate_telemetry(output_path: str = None) -> pd.DataFrame:
    """
    Backward-compatible wrapper for existing scripts.
    Generates telemetry to the standard data path.
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "telemetry_data.csv")

    return generate_synthetic_telemetry(
        num_records_per_satellite=500,
        satellites=["SAT-01", "SAT-02", "SAT-03"],
        anomaly_ratio=0.05,
        random_state=42,
        output_path=output_path
    )


if __name__ == "__main__":
    generate_telemetry()
