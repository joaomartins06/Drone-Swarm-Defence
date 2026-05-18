import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

''' 
Shared data types used across the entire simulator: a 2D point, 
a read-only view into a target state vector, and a radar measurement record. 
No simulation logic lives here.
'''

# State vector convention: [x, vx, y, vy]
# Index 0: x position (m)
# Index 1: x velocity (m/s)
# Index 2: y position (m)
# Index 3: y velocity (m/s)

IX = 0
IVX = 1
IY = 2
IVY = 3
STATE_DIM_CV = 4


class Vec2(NamedTuple):
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def distance_to(self, other: 'Vec2') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return float(np.hypot(dx, dy))
    

@dataclass(frozen=True)
class TargetStateView:
    x: float 
    vx: float
    y: float
    vy: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TargetStateView":
        if arr.shape != (STATE_DIM_CV,):
            raise ValueError(f"Expected array of shape {(STATE_DIM_CV,)}, got {arr.shape}")
            
        return cls(x=arr[IX], vx=arr[IVX], y=arr[IY], vy=arr[IVY])
    
    @property
    def position(self) -> Vec2:
        return Vec2(self.x, self.y)
    
    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))
    
    @property
    def heading_rad(self)-> float:
        return float(np.arctan2(self.vy, self.vx))


@dataclass(frozen=True)
class Measurement:
    #time of the measurement in seconds
    t:float 
    #distance to radar in meters
    range_m: float
    #bearing in radians
    bearing_rad: float
    #radar id
    radar_id: str


    def to_cartesian(self, radar_position: Vec2) -> Vec2:
        #this is to be used for debugging purposes
        #not going to feed it into the KF 
        x = float(radar_position.x + self.range_m * np.cos(self.bearing_rad))
        y = float(radar_position.y + self.range_m * np.sin(self.bearing_rad))
        return Vec2(x, y)
    

