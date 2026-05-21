import numpy as np
 
from sim.core import Clock, Vec2, World
from sim.entities.radar import Radar
from sim.entities.target import Target
from sim.estimation.ekf import EKF
from sim.viz.pygame_renderer import PygameRenderer
 
 
class QuitRequested(Exception):
    """Raised by the render hook when the user closes the window."""
 
 
def main() -> None:
    #set the variables
    dt = 0.1
    measurement_period = 1.0
    render_period = 1.0 / 30.0
    #create a Clock and a World
    clock = Clock(dt=dt, measurement_period=measurement_period, render_period=render_period)
    world = World(clock=clock)
 
    #let's create a list of waypoints for the target to follow
    waypoints = [Vec2(1500.0, 0.0),
                Vec2(1500.0, 1500.0),
                Vec2(-1500.0, 1500.0),
                Vec2(-1500.0, -1500.0),
                Vec2(1500.0, -1500.0),
                Vec2(0.0, 0.0),]
    #create a Target 
    #start it already facing the first waypoint, then it will orient itself (hopefully)
    target = Target(
        target_id="drone1",
        initial_position=Vec2(0.0, 0.0),
        initial_heading=0.0,
        initial_speed=50.0,
        waypoints=waypoints,
        arrival_radius=30.0,
        omega_max=np.deg2rad(30),
        rcs=0.01,
    )
 
    #set the parameters for the radar
    #20m of uncertainty in range seems reasonable 
    sigma_r = 20.0                  
    sigma_b = np.deg2rad(0.5)  
    #create the radar     
    radar = Radar(
        radar_id="radar1",
        position=Vec2(-2000.0, -2000.0), #put it very far way from anything
        target=target,
        sigma_r=sigma_r,
        sigma_b=sigma_b,
        #these were a bit arbitrary and empirically tuned 
        #the next phase will try using real case values for these parameters
        pfa=1e-6,
        C=1e18,
        measurement_period=measurement_period,
        dt=dt,
        seed=42,
    )
 
    #I will start the ekf in the origin with no velocity
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    #but I will set a large initial covariance so it can converge
    initial_covariance = np.diag([sigma_r**2, 1e4, sigma_r**2, 1e4])
    ekf = EKF(
        radar_position=radar.position,
        dt=dt,
        q=1.0,              # process noise spectral density; tune against NEES
        sigma_r=sigma_r,
        sigma_b=sigma_b,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )
 
    #create the renderer
    renderer = PygameRenderer(
        width=1200,
        height=1200,
        world_center=Vec2(0.0, 0.0),
        meters_per_pixel=25,
    )
 
    #add entities to the world
    world.add_entity(target)
    world.add_entity(radar)
    
    #define the callables for the hooks
    #ekf will be every step
    def step_hook(_world: World) -> None:
        ekf.predict()
    
    #radar will be every measurement tick
    def measurement_hook(_world: World) -> None:
        if radar.last_measurement is not None:
            ekf.update(radar.last_measurement)
    
    #renderer will be every render tick
    def render_hook(_world: World) -> None:
        renderer.draw(_world, ekf)
        if renderer.should_quit():
            raise QuitRequested()
 
    #append the hooks to the world
    world.on_step_hooks.append(step_hook)
    world.on_measurement_hooks.append(measurement_hook)
    world.on_render_hooks.append(render_hook)
 
    #run things
    try:
        world.run_for(2500)   # 2 minutes of simulation
    except (QuitRequested, KeyboardInterrupt):
        pass
    finally:
        renderer.close()
 
 
if __name__ == "__main__":
    main()