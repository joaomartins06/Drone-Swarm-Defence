import pytest
from dataclasses import dataclass
 
from sim.core.clock import Clock
from sim.core.world import World
 
 
@dataclass
class _StubEntity:
    """Satisfies the Entity protocol: has entity_id and update()."""
    entity_id: str
    update_count: int = 0
    last_t: float = -1.0
 
    def update(self, t: float, dt: float) -> None:
        self.update_count += 1
        self.last_t = t
 

#clock tests
 
def test_clock_starts_at_zero():
    #check if the clock starts as expected
    c = Clock()
    assert c.t == 0.0
    assert c.tick_count == 0
    assert c.is_measurement_tick()
 
 
def test_clock_time_is_exact_after_many_ticks():
    #check if the clock time is exact after many ticks
    c = Clock(dt=0.1)
    for _ in range(100):
        c.advance()
    assert c.t == pytest.approx(10.0, abs=1e-12)
 
 
def test_clock_measurement_fires_at_correct_ticks():
    #With dt=0.1 and measurement_period=1.0, measurement ticks are 0,10,20,30
    #check if this chekcks out 
    c = Clock(dt=0.1, measurement_period=1.0)
    fired_at = []
    for _ in range(31):
        if c.is_measurement_tick():
            fired_at.append(c.tick_count)
        c.advance()
    assert fired_at == [0, 10, 20, 30]
 
 
def test_clock_rejects_non_positive_dt():
    #check if the clock rejects non-positive dt
    with pytest.raises(ValueError):
        Clock(dt=0.0)
    
    with pytest.raises(ValueError):
        Clock(dt=-0.1)
 
 
def test_clock_rejects_measurement_period_faster_than_dt():
    #check if the clock rejects measurement_period faster than dt
    with pytest.raises(ValueError):
        Clock(dt=0.1, measurement_period=0.05)
 

#world tests

def test_world_steps_entity_correct_number_of_times():
    #run_for(1.0) at dt=0.1 must call update() exactly 10 times.
    w = World(clock=Clock(dt=0.1))
    e = _StubEntity(entity_id="target")
    w.add_entity(e)
    w.run_for(1.0)
    assert e.update_count == 10
 
 
def test_world_entity_receives_correct_final_time():
    #check if the entity created for the tests updates as expected
    w = World(clock=Clock(dt=0.1))
    e = _StubEntity(entity_id="target")
    w.add_entity(e)
    w.run_for(1.0)
    assert e.last_t == pytest.approx(1.0, abs=1e-12)
 
 
def test_world_rejects_duplicate_entity_ids():
    #check if the world rejects duplicate entity IDs
    w = World()
    w.add_entity(_StubEntity(entity_id="e1"))
    with pytest.raises(ValueError):
        w.add_entity(_StubEntity(entity_id="e1"))
 
 
def test_measurement_hook_fires_at_measurement_cadence_not_every_tick():
    #Hook must fire 3 times over 3 seconds, not 30 times (one per physics tick).
    w = World(clock=Clock(dt=0.1, measurement_period=1.0))
    fired_at: list[float] = []
    w.on_measurement_hooks.append(lambda world: fired_at.append(world.clock.t))
    w.run_for(3.0)
    assert len(fired_at) == 3
    assert fired_at == pytest.approx([1.0, 2.0, 3.0], abs=1e-12)


def test_on_step_hook_fires_at_every_tick():
    #On step hook must fire 30 times over 3 seconds (one per physics tick).
    #this is for the predict() step in the EKF
    w = World(clock=Clock(dt=0.1, measurement_period=1.0))
    fired_at: list[float] = []
    w.on_step_hooks.append(lambda world: fired_at.append(world.clock.t))
    w.run_for(3.0)
    assert len(fired_at) == 30
    assert fired_at == pytest.approx([0.1 * i for i in range(1, 31)], abs=1e-12)
