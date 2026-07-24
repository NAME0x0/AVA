"""Governor policy tests — every path with synthetic readings, no hardware."""
from __future__ import annotations

from ratchet.power_governor import Governor, Mode, Reading


def _r(t: float, plugged: bool | None, pct: float | None,
       temp: float | None = 60.0) -> Reading:
    return Reading(t=t, plugged=plugged, battery_pct=pct, gpu_temp_c=temp,
                   gpu_power_w=40.0)


def test_on_battery_pauses() -> None:
    gov = Governor()
    assert gov.decide(_r(0, False, 80)) is Mode.PAUSE


def test_desktop_no_battery_balanced() -> None:
    gov = Governor()
    assert gov.decide(_r(0, None, None)) is Mode.BALANCED


def test_plugged_healthy_balanced() -> None:
    gov = Governor()
    assert gov.decide(_r(0, True, 98)) is Mode.BALANCED


def test_low_battery_trickles_even_plugged() -> None:
    gov = Governor(low_batt=30.0)
    assert gov.decide(_r(0, True, 25)) is Mode.TRICKLE


def test_plugged_draining_trickles_then_recovers() -> None:
    """The user's exact failure: plugged in, GPU load exceeds the brick, battery
    falls. Governor must TRICKLE until the battery recovers past hysteresis."""
    gov = Governor(drain_window_s=600, drain_pct=0.5, recover_pct=1.0)
    assert gov.decide(_r(0, True, 90.0)) is Mode.BALANCED
    assert gov.decide(_r(60, True, 89.8)) is Mode.BALANCED     # under threshold
    assert gov.decide(_r(120, True, 89.4)) is Mode.TRICKLE     # fell 0.6 in window
    assert gov.decide(_r(180, True, 89.5)) is Mode.TRICKLE     # +0.1 only: sticky
    assert gov.decide(_r(240, True, 90.5)) is Mode.BALANCED    # recovered +1.0+


def test_drain_low_water_tracks_further_fall() -> None:
    gov = Governor(drain_window_s=600, drain_pct=0.5, recover_pct=1.0)
    gov.decide(_r(0, True, 90.0))
    assert gov.decide(_r(60, True, 89.0)) is Mode.TRICKLE
    assert gov.decide(_r(120, True, 87.0)) is Mode.TRICKLE     # still falling
    assert gov.decide(_r(180, True, 87.8)) is Mode.TRICKLE     # +0.8 from low water
    assert gov.decide(_r(240, True, 88.1)) is Mode.BALANCED    # +1.1 from low water


def test_thermal_demotes_with_hysteresis() -> None:
    gov = Governor(temp_hi=83, temp_lo=75)
    assert gov.decide(_r(0, True, 95, temp=70)) is Mode.BALANCED
    assert gov.decide(_r(30, True, 95, temp=85)) is Mode.TRICKLE   # hot
    assert gov.decide(_r(60, True, 95, temp=79)) is Mode.TRICKLE   # not cooled enough
    assert gov.decide(_r(90, True, 95, temp=74)) is Mode.BALANCED  # cooled past lo


def test_unplug_clears_drain_state() -> None:
    gov = Governor()
    gov.decide(_r(0, True, 90.0))
    assert gov.decide(_r(60, True, 89.0)) is Mode.TRICKLE
    assert gov.decide(_r(120, False, 88.0)) is Mode.PAUSE      # unplugged
    # replug: drain history cleared, fresh judgement
    assert gov.decide(_r(180, True, 88.0)) is Mode.BALANCED


def test_desktop_thermal_still_respected() -> None:
    gov = Governor(temp_hi=83, temp_lo=75)
    assert gov.decide(_r(0, None, None, temp=90)) is Mode.TRICKLE
    assert gov.decide(_r(30, None, None, temp=70)) is Mode.BALANCED
