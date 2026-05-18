from dataclasses import dataclass, field


''' 
Simulation clock tracking three independent rates: 
physics integration, radar measurement, and display rendering. 
All timing in the simulator derives from this single source of truth.
'''

@dataclass
class Clock:
    dt: float = 0.1
    #how often the radar produces the measurement in seconds
    measurement_period: float = 1.0
    #target display
    render_period : float = 1.0/60.0

    tick_count : int = field(default=0, init=False)

    def __post_init__(self):
        if self.dt  <= 0:
            raise ValueError("dt must be positive")
        if self.measurement_period < self.dt:
            raise ValueError("measurement_period must be larger than time step dt")
        
        self._ticks_per_measurement = max(1, round(self.measurement_period / self.dt))
        self._ticks_per_render = max(1, round(self.render_period / self.dt))

    @property
    def t(self)->float:
        return self.tick_count * self.dt
    
    def advance(self) -> None:
        self.tick_count += 1

    def is_measurement_tick(self) -> bool:
        #check if the current tick is a measurement tick
        return self.tick_count % self._ticks_per_measurement == 0
    
    def is_render_tick(self) -> bool:
        #check if the current tick is a render tick
        return self.tick_count % self._ticks_per_render == 0
    
