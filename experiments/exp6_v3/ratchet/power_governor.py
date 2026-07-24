"""Battery/thermal-aware power governor — Phase 1 of the ratchet.

Why clocks, not power caps: batch-1 LLM decode is memory-bandwidth-bound — SM
cores stall on VRAM — so locking SM clocks low cuts power superlinearly at
near-zero tokens/s cost (arXiv 2605.11999: up to ~32% decode energy recovered;
power caps rarely trigger in decode and are usually locked on laptop GPUs
anyway). Clock locking (`nvidia-smi -lgc`) needs admin on Windows: the nightly
Task-Scheduler job runs elevated; without admin the governor still works via
duty-cycling and pause/resume, it just can't cap clocks (a warning is printed).

The user-observed failure this solves: full-tilt GPU + CPU exceeds the power
brick, so the battery DRAINS while plugged in. Policy: detect plugged-but-
draining and drop to TRICKLE (deep clock cap + duty cycle) until the battery
recovers — the laptop recharges *while the loop keeps running*.

Hardware access is isolated behind read_sensors()/lock_clocks() so every policy
path is unit-testable with synthetic readings.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class Reading:
    t: float                      # time.time()
    plugged: bool | None          # None = no battery present (desktop box)
    battery_pct: float | None
    gpu_temp_c: float | None
    gpu_power_w: float | None


class Mode(str, Enum):
    PAUSE = "PAUSE"          # on battery: never burn the battery
    TRICKLE = "TRICKLE"      # plugged but draining / low / hot: let the brick catch up
    BALANCED = "BALANCED"    # plugged + healthy: capped clocks, continuous work


@dataclass(frozen=True)
class ModeSpec:
    sm_clock_frac: float | None   # fraction of max SM clock to lock to (None = unlocked)
    duty_on_s: int                # work window before pausing
    duty_off_s: int               # cool-off window (0 = continuous)


MODE_SPECS: dict[Mode, ModeSpec] = {
    Mode.PAUSE: ModeSpec(sm_clock_frac=None, duty_on_s=0, duty_off_s=0),
    Mode.TRICKLE: ModeSpec(sm_clock_frac=0.50, duty_on_s=1800, duty_off_s=1800),
    Mode.BALANCED: ModeSpec(sm_clock_frac=0.70, duty_on_s=0, duty_off_s=0),
}


class Governor:
    """Pure policy: feed Readings, get a Mode. Hysteresis on drain + thermal.

    Drain: battery % fell >= drain_pct within drain_window_s while plugged
    -> TRICKLE, sticky until % recovers recover_pct above the low-water mark.
    Thermal: gpu >= temp_hi demotes BALANCED->TRICKLE, sticky until <= temp_lo.
    Low battery (< low_batt) while plugged -> TRICKLE (charge floor).
    No battery hardware (plugged is None) -> BALANCED always.
    """

    def __init__(
        self,
        drain_window_s: float = 600.0,
        drain_pct: float = 0.5,
        recover_pct: float = 1.0,
        low_batt: float = 30.0,
        temp_hi: float = 83.0,
        temp_lo: float = 75.0,
    ) -> None:
        self.drain_window_s = drain_window_s
        self.drain_pct = drain_pct
        self.recover_pct = recover_pct
        self.low_batt = low_batt
        self.temp_hi = temp_hi
        self.temp_lo = temp_lo
        self._hist: deque[tuple[float, float]] = deque()  # (t, battery_pct)
        self._draining = False
        self._drain_low_water: float | None = None
        self._hot = False

    def _update_drain(self, r: Reading) -> None:
        if r.battery_pct is None or not r.plugged:
            # off-charger history is meaningless for plugged-drain detection
            self._hist.clear()
            self._draining = False
            self._drain_low_water = None
            return
        self._hist.append((r.t, r.battery_pct))
        while self._hist and r.t - self._hist[0][0] > self.drain_window_s:
            self._hist.popleft()
        if self._draining:
            self._drain_low_water = min(self._drain_low_water or r.battery_pct, r.battery_pct)
            if r.battery_pct >= (self._drain_low_water + self.recover_pct):
                self._draining = False
                self._drain_low_water = None
        else:
            peak = max(p for _, p in self._hist)
            if peak - r.battery_pct >= self.drain_pct:
                self._draining = True
                self._drain_low_water = r.battery_pct

    def _update_thermal(self, r: Reading) -> None:
        if r.gpu_temp_c is None:
            return
        if r.gpu_temp_c >= self.temp_hi:
            self._hot = True
        elif r.gpu_temp_c <= self.temp_lo:
            self._hot = False

    def decide(self, r: Reading) -> Mode:
        self._update_drain(r)
        self._update_thermal(r)
        if r.plugged is None:                      # desktop / no battery
            return Mode.TRICKLE if self._hot else Mode.BALANCED
        if not r.plugged:
            return Mode.PAUSE
        if r.battery_pct is not None and r.battery_pct < self.low_batt:
            return Mode.TRICKLE
        if self._draining or self._hot:
            return Mode.TRICKLE
        return Mode.BALANCED


# --------------------------------------------------------------------------- hardware IO


def read_sensors() -> Reading:
    plugged: bool | None = None
    pct: float | None = None
    try:
        import psutil

        b = psutil.sensors_battery()
        if b is not None:
            plugged, pct = bool(b.power_plugged), float(b.percent)
    except Exception:  # noqa: BLE001 - sensor loss must never kill the loop
        pass
    temp = power = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        t_s, p_s = [x.strip() for x in out.split(",")]
        temp = float(t_s)
        power = float(p_s) if p_s not in ("[N/A]", "N/A", "") else None
    except Exception:  # noqa: BLE001
        pass
    return Reading(t=time.time(), plugged=plugged, battery_pct=pct,
                   gpu_temp_c=temp, gpu_power_w=power)


def query_max_sm_mhz() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(float(out))
    except Exception:  # noqa: BLE001
        return None


def lock_clocks(frac: float, min_mhz: int = 210) -> bool:
    """Lock SM clocks to [min_mhz, frac*max]. False if denied (needs admin)."""
    mx = query_max_sm_mhz()
    if mx is None:
        return False
    hi = max(min_mhz, int(mx * frac))
    proc = subprocess.run(["nvidia-smi", "-lgc", f"{min_mhz},{hi}"],
                          capture_output=True, text=True, timeout=10)
    return proc.returncode == 0


def reset_clocks() -> bool:
    proc = subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, text=True, timeout=10)
    return proc.returncode == 0


def apply_mode(mode: Mode, can_lock: bool) -> None:
    spec = MODE_SPECS[mode]
    if not can_lock:
        return
    if spec.sm_clock_frac is None:
        reset_clocks()
    else:
        lock_clocks(spec.sm_clock_frac)


# --------------------------------------------------------------------------- watch loop


def watch(interval_s: float = 30.0, apply: bool = False,
          log_path: str | Path | None = None, iterations: int | None = None) -> None:
    """Print/log governor state; with --apply also lock clocks per mode."""
    gov = Governor()
    can_lock = False
    if apply:
        can_lock = lock_clocks(1.0)   # probe permission at full range (harmless)
        if can_lock:
            reset_clocks()
            print("[governor] clock control: OK (elevated)")
        else:
            print("[governor] clock control DENIED - duty-cycle/pause only "
                  "(run elevated for clock caps)")
    last: Mode | None = None
    n = 0
    fh = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        while iterations is None or n < iterations:
            r = read_sensors()
            mode = gov.decide(r)
            if mode is not last:
                print(f"[governor] -> {mode.value}  (batt={r.battery_pct} "
                      f"plugged={r.plugged} temp={r.gpu_temp_c}C power={r.gpu_power_w}W)")
                if apply:
                    apply_mode(mode, can_lock)
                last = mode
            if fh:
                fh.write(json.dumps({"t": r.t, "mode": mode.value,
                                     "battery_pct": r.battery_pct, "plugged": r.plugged,
                                     "gpu_temp_c": r.gpu_temp_c,
                                     "gpu_power_w": r.gpu_power_w}) + "\n")
                fh.flush()
            n += 1
            if iterations is None or n < iterations:
                time.sleep(interval_s)
    finally:
        if fh:
            fh.close()
        if apply and can_lock:
            reset_clocks()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="AVA ratchet power governor")
    ap.add_argument("--watch", action="store_true", help="monitor + print transitions")
    ap.add_argument("--apply", action="store_true", help="also lock clocks per mode (admin)")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--log", type=str, default=None, help="JSONL log path")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()
    if args.watch:
        watch(args.interval, args.apply, args.log, args.iterations)
    else:
        r = read_sensors()
        print(r)
        print("mode:", Governor().decide(r).value)
