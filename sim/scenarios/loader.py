from pathlib import Path
import yaml
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from sim.core import Clock, Vec2, World
from sim.entities.target import Target
from sim.entities.radar import Radar
from sim.estimation.ekf import EKF


class PositionConfig(BaseModel):
    x: float
    y: float


class WorldConfig(BaseModel):
    dt: float = Field(gt=0.0)
    measurement_period: float = Field(gt=0.0)
    render_period: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def check_periods(self) -> "WorldConfig":
        if self.measurement_period < self.dt:
            raise ValueError("measurement_period must be >= dt")
        return self


class TargetConfig(BaseModel):
    target_id: str 
    initial_position: PositionConfig
    initial_heading: float
    initial_speed: float = Field(ge=0.0)
    waypoints: list[PositionConfig]
    arrival_radius: float = Field(gt=0.0)
    omega_max_deg_per_s: float = Field(gt=0.0)
    rcs: float = Field(gt=0.0)


class RadarConfig(BaseModel):
    radar_id: str
    position: PositionConfig
    sigma_r: float = Field(gt=0.0)
    sigma_b_deg: float = Field(gt=0.0)
    pfa: float = Field(gt=0.0, lt=1.0)
    C: float = Field(gt=0.0)
    seed: int | None = None


class EKFConfig(BaseModel):
    initial_state: list[float]
    initial_velocity_variance: float = Field(gt=0.0)
    q: float = Field(gt=0.0)

    @field_validator("initial_state")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("initial_state must have exactly 4 elements")
        return v


class ScenarioConfig(BaseModel):
    world: WorldConfig
    target: TargetConfig
    radar: RadarConfig
    ekf: EKFConfig


def load_scenario(path: str | Path) -> ScenarioConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ScenarioConfig.model_validate(raw)


def build_simulation(config: ScenarioConfig,) -> tuple[World, Target, Radar, EKF]:
    
    #create Clock
    clock = Clock(
        dt=config.world.dt,
        measurement_period=config.world.measurement_period,
        render_period=config.world.render_period,
    )

    #create World using clock
    world = World(clock=clock)
    
    #create Target
    #define the list of waypoints
    waypoints = [Vec2(w.x, w.y) for w in config.target.waypoints]
    target = Target(
        target_id=config.target.target_id,
        initial_position=Vec2(
            config.target.initial_position.x,
            config.target.initial_position.y,
            ),
        initial_heading=config.target.initial_heading,
        initial_speed=config.target.initial_speed,
        waypoints=waypoints,
        arrival_radius=config.target.arrival_radius,
        omega_max=np.deg2rad(config.target.omega_max_deg_per_s),
        rcs=config.target.rcs,
    )
    
    #create Radar
    sigma_b_rad = np.deg2rad(config.radar.sigma_b_deg)
    radar = Radar(
        radar_id=config.radar.radar_id,
        position=Vec2(config.radar.position.x, config.radar.position.y),
        target=target,
        sigma_r=config.radar.sigma_r,
        sigma_b=sigma_b_rad,
        pfa=config.radar.pfa,
        C=config.radar.C,
        measurement_period=config.world.measurement_period,
        dt=config.world.dt,
        seed=config.radar.seed,
    )
    
    #we need to compute the niitial covariance using the radar measurement noise and the initial velocity variance
    #this may be changed, but this is a reasonable choice for our starting covariance matrix
    sigma_r_sq = config.radar.sigma_r ** 2
    initial_cov = np.diag([
        sigma_r_sq,
        config.ekf.initial_velocity_variance,
        sigma_r_sq,
        config.ekf.initial_velocity_variance,
        ])
    #create the EKF
    ekf = EKF(
        radar_position=radar.position,
        dt=config.world.dt,
        q=config.ekf.q,
        sigma_r=config.radar.sigma_r,
        sigma_b=sigma_b_rad,
        initial_state=np.array(config.ekf.initial_state),
        initial_covariance=initial_cov,
    )
    
    #add entities
    #in upcoming phases, I will generalize this to multiple targets and radars
    world.add_entity(target)
    world.add_entity(radar)
    
    return world, target, radar, ekf