import numpy as np
from sim.core import Vec2, Measurement
from sim.entities.target import Target


class Radar:

    def __init__(self, radar_id: str, position: Vec2, target: Target, sigma_r: float,
                 sigma_b: float, pfa: float, C: float, measurement_period: float,
                 dt: float, seed: int | None = None) -> None:
        #id of the radar
        self.entity_id = radar_id
        #position of the radar (fixed, radars do not move)
        self._position = position
        #target to track
        #in upcoming phases of the project we will have multiple targets 
        #and multiple radars
        self.target = target
        #standard deviation of the range measurement noise in meters
        self.sigma_r = sigma_r
        #standard deviation of the bearing measurement noise in radians
        self.sigma_b = sigma_b
        #probability of false alarm (PFA)
        self.pfa = pfa
        #constant C used to compute the detection threshold from the PFA
        #this constant depends on the characteristics of the radar and environment
        self.C = C
        #measurement period in seconds 
        self.measurement_period = measurement_period
        #number of ticks per measurement
        self._ticks_per_measurement = max(1, round(measurement_period / dt))
        #random number generator for noise and false alarms
        self._rng = np.random.default_rng(seed)
        #set the counter to 0
        self._tick_count = 0

        self._last_measurement: Measurement | None = None


    @property
    def position(self) -> Vec2:
        return self._position


    @property
    def last_measurement(self) -> Measurement | None:
        return self._last_measurement
    

    def update(self, t: float, dt: float) -> None:
        self._tick_count += 1

        if self._tick_count % self._ticks_per_measurement != 0:
            return

        #compute the range and the bearing to the target, first without noise
        range_m = self.position.distance_to(self.target.position)
        bearing = np.arctan2(self.target.position.y - self.position.y, 
                             self.target.position.x - self.position.x)
        
        #the tolerance may be changed
        #I will leave it hard coded for now
        if range_m < 1e-9:
            Pd = 1.0
        else: 
            Pd = np.clip(np.power(self.pfa, ((np.power(range_m, 4))/(self.C * self.target.rcs))), 0, 1)

        u = self._rng.uniform()
        if u >= Pd:
            #missed detection, do not return a measurement
            self._last_measurement = None
            return
        
        #compute the noisy measurements
        range_noisy = range_m + self._rng.normal(0, self.sigma_r)
        bearing_noisy = bearing + self._rng.normal(0, self.sigma_b)
        #just to enforce that it is in the interval [-pi, pi]
        bearing_noisy = np.arctan2(np.sin(bearing_noisy), np.cos(bearing_noisy))

        measurement = Measurement(t=t, range_m=range_noisy, bearing_rad=bearing_noisy, 
                                  radar_id=self.entity_id)
        
        self._last_measurement = measurement

