#!/usr/bin/env python3
"""GPU watchdog monitor for kraken-stark benchmarking.

Polls nvidia-smi at 1-second intervals, logs everything to CSV,
and prints real-time alerts for anomalies: throttling, temp spikes,
clock drops, power events, or driver failures (GPU gone AWOL).

Usage:
    python gpu_monitor.py                  # monitor to stdout + gpu_log_<timestamp>.csv
    python gpu_monitor.py --interval 0.5   # poll every 500ms
    python gpu_monitor.py --quiet          # CSV only, no live console output

Log survives crashes: every row is flushed immediately.
"""

import subprocess
import sys
import time
import os
import signal
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

NVIDIA_SMI = "nvidia-smi"

QUERY_FIELDS = [
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.memory",
    "clocks.max.graphics",
    "clocks.max.memory",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "pstate",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.gpu_idle",
]

FIELD_NAMES = [
    "temp_c",
    "power_w",
    "power_limit_w",
    "clk_gfx_mhz",
    "clk_mem_mhz",
    "clk_gfx_max_mhz",
    "clk_mem_max_mhz",
    "util_gpu_pct",
    "util_mem_pct",
    "vram_used_mib",
    "vram_total_mib",
    "pstate",
    "hw_slowdown",
    "hw_thermal",
    "hw_power_brake",
    "sw_thermal",
    "sw_power_cap",
    "gpu_idle",
]

# Thresholds
TEMP_WARN_C = 85
TEMP_CRIT_C = 90
POWER_WARN_PCT = 0.95  # warn at 95% of power limit
CLK_DROP_PCT = 0.20     # alert if gfx clock drops >20% below max

# ── Helpers ─────────────────────────────────────────────────────────────────

def query_gpu():
    """Run nvidia-smi and return parsed field dict, or None on failure."""
    cmd = [
        NVIDIA_SMI,
        "--query-gpu=" + ",".join(QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5
        )
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi TIMEOUT (5s) — GPU may be hung"
    except FileNotFoundError:
        return None, "nvidia-smi not found"
    except Exception as e:
        return None, f"nvidia-smi error: {e}"

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return None, f"nvidia-smi exit code {result.returncode}: {stderr}"

    raw = result.stdout.strip()
    if not raw:
        return None, "nvidia-smi returned empty output"

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != len(FIELD_NAMES):
        return None, f"nvidia-smi field count mismatch: got {len(parts)}, expected {len(FIELD_NAMES)}"

    data = {}
    for name, val in zip(FIELD_NAMES, parts):
        # Try to convert numeric fields
        try:
            if "." in val:
                data[name] = float(val)
            else:
                data[name] = int(val)
        except (ValueError, TypeError):
            data[name] = val
    return data, None


def check_anomalies(data, prev_data):
    """Return list of alert strings for this sample."""
    alerts = []

    temp = data.get("temp_c", 0)
    if isinstance(temp, (int, float)):
        if temp >= TEMP_CRIT_C:
            alerts.append(f"CRITICAL TEMP: {temp}C (>={TEMP_CRIT_C}C)")
        elif temp >= TEMP_WARN_C:
            alerts.append(f"HIGH TEMP: {temp}C (>={TEMP_WARN_C}C)")

    power = data.get("power_w", 0)
    limit = data.get("power_limit_w", 0)
    if isinstance(power, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
        if power >= limit * POWER_WARN_PCT:
            alerts.append(f"POWER LIMIT: {power:.1f}W / {limit:.1f}W ({power/limit*100:.0f}%)")

    gfx = data.get("clk_gfx_mhz", 0)
    gfx_max = data.get("clk_gfx_max_mhz", 0)
    if isinstance(gfx, (int, float)) and isinstance(gfx_max, (int, float)) and gfx_max > 0:
        # Only alert on clock drop if GPU is actually working (not idle)
        idle = data.get("gpu_idle", "")
        util = data.get("util_gpu_pct", 0)
        if str(idle) == "Not Active" and isinstance(util, (int, float)) and util > 10:
            drop_pct = 1.0 - (gfx / gfx_max)
            if drop_pct > CLK_DROP_PCT:
                alerts.append(f"CLOCK DROP: {gfx}MHz / {gfx_max}MHz max ({drop_pct*100:.0f}% below max)")

    # Throttle reasons
    for field, label in [
        ("hw_slowdown", "HW SLOWDOWN"),
        ("hw_thermal", "HW THERMAL SLOWDOWN"),
        ("hw_power_brake", "HW POWER BRAKE"),
        ("sw_thermal", "SW THERMAL THROTTLE"),
        ("sw_power_cap", "SW POWER CAP"),
    ]:
        val = str(data.get(field, "Not Active"))
        if val != "Not Active":
            alerts.append(f"THROTTLE: {label} = {val}")

    # VRAM pressure
    used = data.get("vram_used_mib", 0)
    total = data.get("vram_total_mib", 1)
    if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
        pct = used / total
        if pct > 0.95:
            alerts.append(f"VRAM CRITICAL: {used:.0f}/{total:.0f} MiB ({pct*100:.1f}%)")
        elif pct > 0.85:
            alerts.append(f"VRAM HIGH: {used:.0f}/{total:.0f} MiB ({pct*100:.1f}%)")

    return alerts


def format_console_line(elapsed, data, alerts):
    """One-line console status."""
    temp = data.get("temp_c", "?")
    power = data.get("power_w", "?")
    limit = data.get("power_limit_w", "?")
    gfx = data.get("clk_gfx_mhz", "?")
    mem = data.get("clk_mem_mhz", "?")
    util = data.get("util_gpu_pct", "?")
    vram = data.get("vram_used_mib", "?")
    vtot = data.get("vram_total_mib", "?")
    pst = data.get("pstate", "?")

    line = (
        f"[{elapsed:>7.1f}s] "
        f"{temp:>3}C  "
        f"{power:>6}W/{limit}W  "
        f"GFX {gfx:>5}MHz  "
        f"MEM {mem:>5}MHz  "
        f"GPU {util:>3}%  "
        f"VRAM {vram:>6}/{vtot} MiB  "
        f"{pst}"
    )
    if alerts:
        line += "  *** " + " | ".join(alerts)
    return line


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU watchdog monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds (default: 1.0)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output (CSV log only)")
    parser.add_argument("--logdir", type=str, default=".", help="Directory for log files (default: cwd)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.logdir) / f"gpu_log_{timestamp}.csv"

    # Track stats for summary
    stats = {
        "max_temp": 0, "max_power": 0, "max_vram": 0,
        "min_gfx_clk": 999999, "max_gfx_clk": 0,
        "total_alerts": 0, "samples": 0, "failures": 0,
        "throttle_seconds": 0,
    }

    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"GPU Monitor started — logging to {log_path}")
    print(f"Poll interval: {args.interval}s | Ctrl+C to stop\n")

    with open(log_path, "w", buffering=1) as f:  # line-buffered
        # CSV header
        header = "timestamp,elapsed_s," + ",".join(FIELD_NAMES) + ",alerts"
        f.write(header + "\n")
        f.flush()

        t_start = time.monotonic()
        prev_data = None
        consecutive_failures = 0

        while running:
            t_sample = time.monotonic()
            elapsed = t_sample - t_start
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            data, error = query_gpu()

            if error:
                consecutive_failures += 1
                stats["failures"] += 1
                alert_str = f"FAILURE({consecutive_failures}): {error}"

                f.write(f"{now},{elapsed:.3f}," + ",".join(["ERR"] * len(FIELD_NAMES)) + f",\"{alert_str}\"\n")
                f.flush()
                os.fsync(f.fileno())  # force to disk — we might be about to crash

                if not args.quiet:
                    print(f"[{elapsed:>7.1f}s] *** {alert_str}")

                if consecutive_failures >= 5:
                    print("\n*** 5 consecutive nvidia-smi failures — GPU is AWOL ***")
                    print(f"*** Last successful sample was {consecutive_failures * args.interval:.1f}s ago ***")

                prev_data = None
            else:
                consecutive_failures = 0
                stats["samples"] += 1

                alerts = check_anomalies(data, prev_data)
                stats["total_alerts"] += len(alerts)

                # Track throttle seconds (any non-idle throttle active)
                any_throttle = any(
                    str(data.get(f, "Not Active")) != "Not Active"
                    for f in ["hw_slowdown", "hw_thermal", "hw_power_brake", "sw_thermal", "sw_power_cap"]
                )
                if any_throttle:
                    stats["throttle_seconds"] += args.interval

                # Update peak stats
                t = data.get("temp_c", 0)
                if isinstance(t, (int, float)):
                    stats["max_temp"] = max(stats["max_temp"], t)
                p = data.get("power_w", 0)
                if isinstance(p, (int, float)):
                    stats["max_power"] = max(stats["max_power"], p)
                v = data.get("vram_used_mib", 0)
                if isinstance(v, (int, float)):
                    stats["max_vram"] = max(stats["max_vram"], v)
                g = data.get("clk_gfx_mhz", 0)
                if isinstance(g, (int, float)) and g > 0:
                    u = data.get("util_gpu_pct", 0)
                    if isinstance(u, (int, float)) and u > 10:
                        stats["min_gfx_clk"] = min(stats["min_gfx_clk"], g)
                        stats["max_gfx_clk"] = max(stats["max_gfx_clk"], g)

                # Write CSV row
                vals = []
                for name in FIELD_NAMES:
                    v = data.get(name, "")
                    vals.append(str(v))
                alert_str = "|".join(alerts) if alerts else ""
                f.write(f"{now},{elapsed:.3f}," + ",".join(vals) + f",\"{alert_str}\"\n")
                f.flush()

                if not args.quiet:
                    print(format_console_line(elapsed, data, alerts))

                prev_data = data

            # Sleep remainder of interval
            dt = time.monotonic() - t_sample
            if dt < args.interval:
                time.sleep(args.interval - dt)

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed_total = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print(f"GPU Monitor Summary ({elapsed_total:.1f}s)")
    print(f"{'='*60}")
    print(f"  Samples:          {stats['samples']}")
    print(f"  Failures:         {stats['failures']}")
    print(f"  Total alerts:     {stats['total_alerts']}")
    print(f"  Max temp:         {stats['max_temp']}C")
    print(f"  Max power:        {stats['max_power']:.1f}W")
    print(f"  Max VRAM:         {stats['max_vram']:.0f} MiB")
    if stats["min_gfx_clk"] < 999999:
        print(f"  GFX clock range:  {stats['min_gfx_clk']}-{stats['max_gfx_clk']} MHz")
    print(f"  Throttle time:    {stats['throttle_seconds']:.1f}s")
    print(f"  Log:              {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
