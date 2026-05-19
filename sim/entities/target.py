import numpy as np
from sim.core import Vec2, IW, IX, IY, IVX, IVY
from sim.core.types import TargetStateView

''' 
This class represents the target of the simulation, which will be our drones
The target is represented by a state vector of dimension 5:
[ x, vx, y, vy, omega ]
where:
- x and y are the position of the target in meters
- vx and vy are the velocity of the target in meters per second
- omega is the angular velocity of the target in radians per second, which is used to turn

Then we also specify a list of waypoints that the target will follow, 
a radius in meters to consider the target has arrived at a waypoint and ´
the maximum angular velocity in radians per second when turning (basically, how agile is the drone)

The main method is update, which computes the next state of the drone given its current state, 
time and timestep and its next waypoint.
'''

class Target:

    def __init__(self, target_id: str, initial_position: Vec2, initial_heading: float, 
                 initial_speed: float, waypoints: list[Vec2], arrival_radius:float, 
                 omega_max: float, rcs:float = 0.01 ) -> None:

        #target's id
        self.entity_id = target_id

        #compute the state vector
        self.state = np.array([initial_position.x, initial_speed * np.cos(initial_heading), 
                                       initial_position.y, initial_speed * np.sin(initial_heading), 0.0])
        
        self.waypoints = waypoints
        #radius in meters to consider the target has arrived at a waypoint
        self.arrival_radius = arrival_radius
        #maximum angular velocity in radians per second when turning
        #the absolute is just to ensure
        if omega_max < 0:
            raise ValueError(f"omega_max must be non-negative, got {omega_max}")
        
        self.omega_max = omega_max

        #radar cross section in m^2
        #a typical value for a small drone is around 0.01 m^2
        #naturally it varies depending on the stealth
        #this could even be a B2 stealth bomber with a radar cross section of 0.0001 m^2
        #but let's stick to the drones haha
        self.rcs = rcs



    @property
    def position(self) -> Vec2:
        return Vec2(self.state[IX], self.state[IY])
    
    @property
    def state_view(self) -> TargetStateView:
        return TargetStateView.from_array(self.state[:4])
    

    def _compute_and_update_omega(self, dt: float) -> float:

        if not self.waypoints:
            self.state[IW] = 0.0
            return 0.0

        #compute heading from the drone
        heading_rad = np.arctan2(self.state[IVY], self.state[IVX])
        #copmute the bearing to the next waypoint
        bearing_to_waypoint_rad = np.arctan2(self.waypoints[0].y - self.state[IY], 
                                             self.waypoints[0].x - self.state[IX])
        #compute the error between the two
        error_rad = bearing_to_waypoint_rad - heading_rad
        #make sure it is in [-pi, pi]
        error_rad = np.arctan2(np.sin(error_rad), np.cos(error_rad))

        self.state[IW] = np.clip(error_rad / dt, -self.omega_max, self.omega_max)

        return self.state[IW]
    

    def _update_state(self, dt: float, omega: float, tolerance:float = 1e-6) -> None:
        ''' 
        Recall that:
        x  ← x  + sin(ω·dt)/ω · vx  -  (1 - cos(ω·dt))/ω · vy
        y  ← y  + (1 - cos(ω·dt))/ω · vx  +  sin(ω·dt)/ω · vy
        vx ← cos(ω·dt) · vx  -  sin(ω·dt) · vy
        vy ← sin(ω·dt) · vx  +  cos(ω·dt) · vy

        I chose to compute the exact solution, giving a tolerance for when omega is close to zero,
        so it does not diverge
        If I used the euler method it could diverge
        Runge-Kutta was also an option, but as applying the exact solution is more efficient and more 
        accurate
        '''
        if np.abs(omega) < tolerance:
            #vx and vy stay the same
            self.state[IX] = self.state[IX] + self.state[IVX] * dt
            self.state[IY] = self.state[IY] + self.state[IVY] * dt
        else:
            vx = self.state[IVX]
            vy = self.state[IVY]
            #update positions
            self.state[IX] = self.state[IX] + vx * np.sin(omega * dt)/omega \
                - vy * (1 - np.cos(omega * dt)) / omega
            self.state[IY] = self.state[IY] + vx * (1 - np.cos(omega * dt)) / omega \
                + vy * np.sin(omega * dt) / omega
            #update velocities
            self.state[IVX] = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
            self.state[IVY] = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
            

    def update(self, t: float, dt: float) -> None:
        #compute the error between the heading and the bearing to the next waypoint
        #and update omega
        omega = self._compute_and_update_omega(dt)
        #update the state of the target
        self._update_state(dt, omega)
        #check if we have arrived at the next waypoint and update the list
        self._distance_to_waypoint()


    def _distance_to_waypoint(self) -> None:
        #compute the distance to the next waypoint
        if not self.waypoints:
            return
        
        distance = self.position.distance_to(self.waypoints[0])
        if distance < self.arrival_radius:
            #remove the waypoint from the list
            self.waypoints.pop(0)
        
            
