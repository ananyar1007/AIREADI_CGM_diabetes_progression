import numpy as np
import datetime
import pandas as pd

import pandas as pd
from datetime import time

def get_aligned_df(df, time_col="Timestamp"):
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Find the first timestamp that falls between midnight and 5 min after midnight
    mask = (df[time_col].dt.time >= time(0, 0)) & (df[time_col].dt.time <= time(0, 5))
    if mask.any():
        first_valid_time = df.loc[mask, time_col].iloc[0]
        # Keep everything from that time onward
        df = df[df[time_col] >= first_valid_time]
    else:
        # If no midnight window found, keep DataFrame as is (or drop entirely)
        print(f"⚠️ No midnight window found in DataFrame, skipping...")
    return df

def convert_time_string_to_datetime(t_str):
    """Converts time string to datetime format. Does not convert to local time.
    Args:
        t_str (str): UTC time string such as 2023-08-01T20:39:33Z
    Returns: datetime object
    """
    datetime_object = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ")  # 4 digit Year
    return datetime_object



def _detect_spikes_with_resolution(times, values, window=18, threshold=30, require_resolved=True):
    """
    Detect spikes and compute resolution. Optionally require the spike to resolve.
    Resolution = time from peak to first time value <= baseline + 0.5*amplitude.
    Baseline = max(left_min, right_min) over `window` samples on each side.
    """
    n = len(values)
    out = []

    for i in range(window, n - window):
        vi = values[i]
        if vi > values[i-1] and vi > values[i+1]:  # local max
            left_min  = float(np.min(values[i-window:i]))
            right_min = float(np.min(values[i+1:i+1+window]))
            baseline  = max(left_min, right_min)
            amp = float(vi - baseline)
            if amp >= threshold:
                half_target = baseline + 0.5 * amp

                # scan forward to find first crossing
                hit_idx = None
                for j in range(i+1, n):
                    if values[j] <= half_target:
                        hit_idx = j
                        break

                if hit_idx is None and require_resolved:
                    continue  # skip unresolved spikes entirely

                if hit_idx is not None and pd.notna(times.iloc[i]) and pd.notna(times.iloc[hit_idx]):
                    dt = times.iloc[hit_idx] - times.iloc[i]
                    dt_min = dt.total_seconds() / 60.0
                else:
                    dt = pd.NaT
                    dt_min = np.nan

                out.append({
                    "peak_idx": i,
                    "peak_time": times.iloc[i],
                    "peak_value": float(vi),
                    "baseline": float(baseline),
                    "amplitude": float(amp),
                    "half_target": float(half_target),
                    "resolution_timedelta": dt,
                    "resolution_minutes": dt_min,
                })

    return pd.DataFrame(out)

def compute_spike_metrics(
    df,
    time_col="start_dtime",
    value_col="blood_glucose_value",
    window=18,
    threshold=30,
):
    """
    Computes:
      - avg_spike_resolution_min: mean time (min) to absorb 50% of each spike (resolved spikes only)
      - expected_daily_spikes: mean spikes/day (resolved spikes only)
      - mean_glucose: mean glucose over entire tracking period
      - expected_max_spike_relative_value: avg over days of (daily_max - baseline)/baseline,
        baseline = overall mean glucose
      - hyper_time_pct: % of readings > 150 mg/dL
      - nocturnal_hypoglycemia: avg over days of the minimum glucose between 00:00 and 07:00
    """

    # Prepare time series
    s = df.copy()
    s[time_col] = pd.to_datetime(s[time_col], errors="coerce")
    s = s.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if s.empty:
        return {
            "avg_spike_resolution_min": np.nan,
            "expected_daily_spikes": np.nan,
            "mean_glucose": np.nan,
            "expected_max_spike_relative_value": np.nan,
            "hyper_time_pct": np.nan,
            "nocturnal_hypoglycemia": np.nan
        }

    times = s[time_col]
    values = s[value_col].to_numpy()

    # ---- Spike metrics ----
    spike_df = _detect_spikes_with_resolution(
        times=times,
        values=values,
        window=window,
        threshold=threshold,
        require_resolved=True,
    )
    if not spike_df.empty:
        spike_df = spike_df.dropna(
            subset=["peak_time", "resolution_minutes", "resolution_timedelta"]
        )
        spike_df = spike_df[np.isfinite(spike_df["resolution_minutes"])]

    if not spike_df.empty:
        avg_spike_resolution_min = float(spike_df["resolution_minutes"].mean())
        spike_times = spike_df["peak_time"]
        spikes_per_day = spike_times.groupby(spike_times.dt.date).count()
        expected_daily_spikes = float(spikes_per_day.mean()) if not spikes_per_day.empty else np.nan
    else:
        avg_spike_resolution_min = np.nan
        expected_daily_spikes = np.nan

    # ---- Global glucose metrics ----
    mean_glucose = float(s[value_col].mean())
    hyper_time_pct = float((s[value_col] > 150).mean() * 100.0)

    # Expected maximum spike relative value, baseline = overall mean
    baseline = mean_glucose
    if np.isfinite(baseline) and baseline != 0.0:
        daily_max = s.groupby(times.dt.date)[value_col].max()
        expected_max_spike_relative_value = float(((daily_max - baseline) / baseline).mean())
    else:
        expected_max_spike_relative_value = np.nan

    # ---- Nocturnal hypoglycemia: avg(min between 00:00 and 07:00) across days ----
    nocturnal_mask = times.dt.hour.between(0, 6)  # 00:00–06:59
    nocturnal = s.loc[nocturnal_mask, [time_col, value_col]]
    if not nocturnal.empty:
        nocturnal_min_per_day = nocturnal.groupby(nocturnal[time_col].dt.date)[value_col].min()
        nocturnal_hypoglycemia = float(nocturnal_min_per_day.mean()) if not nocturnal_min_per_day.empty else np.nan
    else:
        nocturnal_hypoglycemia = np.nan

    return {
        "avg_spike_resolution_min": avg_spike_resolution_min,
        "expected_daily_spikes": expected_daily_spikes,
        "mean_glucose": mean_glucose,
        "expected_max_spike_relative_value": expected_max_spike_relative_value,  # unitless ratio
        "hyper_time_pct": hyper_time_pct,
        "nocturnal_hypoglycemia": nocturnal_hypoglycemia
    }
