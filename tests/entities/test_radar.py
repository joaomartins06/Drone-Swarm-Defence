import numpy as np
import pytest
 
from sim.core import Vec2
from sim.core.entity import Entity
from sim.entities.radar import Radar
from sim.entities.target import Target
 
 
def make_stationary_target(position: Vec2 = Vec2(100.0, 0.0), rcs: float = 0.01) -> Target:
    #create a simple target with zero speed. So it is just a point haha
    return Target(
        target_id="t1",
        initial_position=position,
        initial_heading=0.0,
        initial_speed=0.0,
        waypoints=[],
        arrival_radius=10.0,
        omega_max=np.deg2rad(30),
        rcs=rcs,
    )
 
 
def make_radar(
    target: Target,
    position: Vec2 = Vec2(0.0, 0.0),
    sigma_r: float = 10.0,
    sigma_b: float = np.deg2rad(0.5),
    pfa: float = 1e-6,
    C: float = 1e16,   # very high — ensures Pd ~ 1 at close range
    measurement_period: float = 1.0,
    dt: float = 0.1,
    seed: int = 42,
) -> Radar:
    #create a radar with the given parameters, tracking our stationary point 
    return Radar(
        radar_id="r1",
        position=position,
        target=target,
        sigma_r=sigma_r,
        sigma_b=sigma_b,
        pfa=pfa,
        C=C,
        measurement_period=measurement_period,
        dt=dt,
        seed=seed,
    )
 
 
#tests
 
def test_radar_satisfies_entity_protocol():
    #check if radar fulfils the protocol
    target = make_stationary_target()
    radar = make_radar(target)
    assert isinstance(radar, Entity)
 
 
def test_no_measurement_on_non_measurement_ticks():
    #check it there is no measurement on non-measurement ticks
    target = make_stationary_target()
    radar = make_radar(target)
    # First 9 ticks: not measurement ticks.
    for _ in range(9):
        radar.update(0.0, 0.1)
        assert radar.last_measurement is None
 
 
def test_measurement_produced_at_close_range():
    #check if the radar measures something
    target = make_stationary_target(position=Vec2(10.0, 0.0))
    radar = make_radar(target, C=1e20)  #this is absurdly high, just to make sure that P is 1
    #advance to first click 
    for _ in range(10):
        radar.update(0.0, 0.1)

    assert radar.last_measurement is not None
 
 
def test_no_measurement_at_extreme_range():
    #test for extremely large distances if it produces smth (it shouldnt)
    target = make_stationary_target(position=Vec2(1e8, 0.0))  #100.000km hahaha (earth perimeter ~40.000km)
    radar = make_radar(target, C=1.0)   # small C
    #this way, pd is basically 0
    #1000 measurement ticks equiv 10,000 physics ticks.
    for _ in range(10_000):
        radar.update(0.0, 0.1)
        #a measurement can happen, it's just extremeeeeeely unlikely
        assert radar.last_measurement is None


def test_no_nan_in_measurement():
    #check if for a lot of measurements is ever produced a NaN (it shouldnt)
    target = make_stationary_target(position=Vec2(500.0, 500.0))
    radar = make_radar(target, C=1e20)
    for _ in range(1000):
        radar.update(0.0, 0.1)
        m = radar.last_measurement
        if m is not None:
            assert np.isfinite(m.range_m)
            assert np.isfinite(m.bearing_rad)
 
 
def test_noise_is_applied_to_measurement():
    #check if the noise is doing his thing
    target = make_stationary_target(position=Vec2(100.0, 0.0))
    radar = make_radar(target, C=1e20, sigma_r=10.0, sigma_b=np.deg2rad(1.0)) #again, very high C
    for _ in range(10):
        radar.update(0.0, 0.1)
    m = radar.last_measurement
    #should not be none, like in the previous test
    assert m is not None
    #should differ due to noise
    assert m.range_m != 100.0
    assert m.bearing_rad != 0.0
 
 
def test_measurement_is_close_to_truth_in_expectation():
    #same thing as before, but the measurement should be close to the truth, on average
    target = make_stationary_target(position=Vec2(1000.0, 0.0))
    radar = make_radar(target, C=1e25, sigma_r=10.0, sigma_b=np.deg2rad(0.5))
    ranges = []
    bearings = []
    # 500 measurement ticks.
    for _ in range(5000):
        radar.update(0.0, 0.1)
        if radar.last_measurement is not None:
            ranges.append(radar.last_measurement.range_m)
            bearings.append(radar.last_measurement.bearing_rad)
    
    assert len(ranges) > 100 #we should have enough samples to get a faithful extimate
    assert np.mean(ranges) == pytest.approx(1000.0, abs=5.0)
    assert np.mean(bearings) == pytest.approx(0.0, abs=0.01)
 
 
def test_bearing_is_wrapped_to_pi_interval():
    #target in the west, check if the bearing is wrapped to the interval [-pi, pi]
    target = make_stationary_target(position=Vec2(-1000.0, 0.0))
    radar = make_radar(target, C=1e25, sigma_b=np.deg2rad(2.0), seed=0)
    for _ in range(50):
        radar.update(0.0, 0.1)

        if radar.last_measurement is not None:
            b = radar.last_measurement.bearing_rad
            assert -np.pi <= b <= np.pi
 
 
def test_seed_reproducibility():
    #check reprocuibility by using the same seed and parameters
    target1 = make_stationary_target(position=Vec2(200.0, 100.0))
    target2 = make_stationary_target(position=Vec2(200.0, 100.0))
    radar1 = make_radar(target1, C=1e20, seed=123)
    radar2 = make_radar(target2, C=1e20, seed=123)
    for _ in range(100):
        radar1.update(0.0, 0.1)
        radar2.update(0.0, 0.1)
    m1 = radar1.last_measurement
    m2 = radar2.last_measurement
    assert m1 is not None and m2 is not None
    assert m1.range_m == m2.range_m
    assert m1.bearing_rad == m2.bearing_rad