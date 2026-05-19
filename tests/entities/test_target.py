import numpy as np
import pytest
 
from sim.core import Vec2
from sim.core.entity import Entity
from sim.entities.target import Target
 
 
 #create an object Target
def make_target(
    position: Vec2 = Vec2(0.0, 0.0),
    heading: float = 0.0,
    speed: float = 100.0,
    waypoints: list[Vec2] | None = None,
    arrival_radius: float = 50.0,
    omega_max: float = np.deg2rad(30),
) -> Target:
    return Target(
        target_id="t1",
        initial_position=position,
        initial_heading=heading,
        initial_speed=speed,
        waypoints=waypoints if waypoints is not None else [],
        arrival_radius=arrival_radius,
        omega_max=omega_max,
    )
 

#tests
 
def test_target_satisfies_entity_protocol():
    #check if it satisfies the Entity protocol
    t = make_target()
    assert isinstance(t, Entity)
 
 
def test_straight_line_flight_position():
    #check if the target moves as it is expected for a simple case
    t = make_target(speed=100.0, heading=0.0, waypoints=[])
    dt = 0.1
    for _ in range(10):
        t.update(0.0, dt)
    assert t.state[0] == pytest.approx(100.0, abs=1e-9)   # x
    assert t.state[2] == pytest.approx(0.0, abs=1e-9)     # y
 
 
def test_straight_line_flight_speed_unchanged():
    #check if the speed remains constant during straight-line flight
    t = make_target(speed=100.0, heading=0.0, waypoints=[])
    initial_speed = np.hypot(t.state[1], t.state[3])
    for _ in range(10):
        t.update(0.0, 0.1)
    final_speed = np.hypot(t.state[1], t.state[3])
    assert final_speed == pytest.approx(initial_speed, abs=1e-9)
 
 
def test_speed_conserved_during_turn():
    #check if the speed (up to some tolerance) is conserved during a turn
    waypoint = Vec2(0.0, 1000.0)   #it's in the north, which forces it to turn left
    t = make_target(speed=100.0, heading=0.0, waypoints=[waypoint], arrival_radius=10.0)
    initial_speed = np.hypot(t.state[1], t.state[3])
    #turn for 5 second in direction of the waypoint
    for _ in range(50):
        t.update(0.0, 0.1)
    final_speed = np.hypot(t.state[1], t.state[3])
    assert final_speed == pytest.approx(initial_speed, abs=1e-6)
 
 
def test_waypoint_consumed_on_arrival():
    #check if the target consumes the waypoint when it arrives within the arrival radius
    waypoint = Vec2(5.0, 0.0) #a waypoint that is close from the drone
    t = make_target(
        position=Vec2(0.0, 0.0),
        heading=0.0,
        speed=100.0,
        waypoints=[waypoint],
        arrival_radius=50.0,
    )
    # after on update, the drone should have moved 10 meters,
    #which is inside the arrival radius, so the waypoint should be consumed
    t.update(0.0, 0.1)
    assert len(t.waypoints) == 0
 
 
def test_no_crash_or_nan_on_empty_waypoints():
    #check if the tarhet does not crash or produces NaN when there are no waypoints
    #should just continue in a straight line with the same velocity
    t = make_target(waypoints=[])
    t.update(0.0, 0.1)
    assert not np.any(np.isnan(t.state))
    assert not np.any(np.isinf(t.state))
 
 
def test_heading_wrap_short_turn():
    #check if teh angle wrapping works
    heading = np.pi - 0.1
    t = make_target(
        position=Vec2(0.0, 0.0),
        heading=heading,
        speed=50.0,
        waypoints=[Vec2(-1000.0, -10.0)], #the waypoint is basically south (slightly west)
        arrival_radius=20.0,
        omega_max=np.deg2rad(45),
    )
 
    def heading_error(target: Target) -> float:
        current = np.arctan2(target.state[3], target.state[1])
        desired = np.arctan2(
            target.waypoints[0].y - target.state[2],
            target.waypoints[0].x - target.state[0],
        )
        err = desired - current
        return float(np.arctan2(np.sin(err), np.cos(err)))
 
    initial_error = abs(heading_error(t))
    for _ in range(10):
        t.update(0.0, 0.1)
    final_error = abs(heading_error(t))
 
    assert final_error < initial_error