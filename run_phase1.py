import sys

from sim.core import Clock, Vec2, World
from sim.entities.radar import Radar
from sim.entities.target import Target
from sim.estimation.ekf import EKF
from sim.viz.pygame_renderer import PygameRenderer
from sim.scenarios.loader import build_simulation, load_scenario
 
 
class QuitRequested(Exception):
    """Raised by the render hook when the user closes the window."""
 
 
def main() -> None:
    #get the path to scenario
    scenario_path = (
        sys.argv[1] if len(sys.argv) > 1 else "sim/scenarios/phase1_default.yaml"
    )
 
    #create the config
    config = load_scenario(scenario_path)
    #and the simulation
    world, target, radar, ekf = build_simulation(config)
 
    #just defining the renderer params
    renderer = PygameRenderer(
        width=1200,
        height=1200,
        world_center=Vec2(0.0, 0.0),
        meters_per_pixel=25,
    )
 
    #define ekf.predict as a step hook (every tick)
    def step_hook(_world: World) -> None:
        ekf.predict()
    
    #define ekf.update as a measurement hook (every measurement tick), using the last measurement from the radar
    def measurement_hook(_world: World) -> None:
        if radar.last_measurement is not None:
            ekf.update(radar.last_measurement)
 
    #define the render hook (every render tick)
    def render_hook(_world: World) -> None:
        renderer.draw(_world, ekf)
        if renderer.should_quit():
            raise QuitRequested()
 
    #add hooks accordingly
    world.on_step_hooks.append(step_hook)
    world.on_measurement_hooks.append(measurement_hook)
    world.on_render_hooks.append(render_hook)
 

    #run the simulation
    try:
        world.run_for(config.world.duration_s)
    except (QuitRequested, KeyboardInterrupt):
        pass
    finally:
        renderer.close()
 
 
if __name__ == "__main__":
    main()