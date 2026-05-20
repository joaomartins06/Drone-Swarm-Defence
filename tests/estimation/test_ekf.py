import numpy as np
import pytest
 
from sim.core import Measurement, Vec2
from sim.estimation.ekf import EKF
 

#aux function to create an EKF filter 
def make_ekf(
    radar_position: Vec2 = Vec2(0.0, 0.0),
    dt: float = 1.0,
    q: float = 1.0,
    sigma_r: float = 10.0,
    sigma_b: float = np.deg2rad(0.5),
    initial_state: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
) -> EKF:
    if initial_state is None:
        initial_state = np.array([100.0, 5.0, 100.0, 3.0])
    if initial_covariance is None:
        initial_covariance = np.diag([100.0, 100.0, 100.0, 100.0])
    return EKF(
        radar_position=radar_position,
        dt=dt,
        q=q,
        sigma_r=sigma_r,
        sigma_b=sigma_b,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )
 
 
#test the predict step from the EKF 
def test_predict_propagates_position_via_velocity():
    #check if the precict step correctly propagates the position for a very simple case
    ekf = make_ekf(
        dt=1.0,
        initial_state=np.array([0.0, 10.0, 0.0, 5.0]),
        initial_covariance=np.eye(4),
    )
    ekf.predict()
    expected = np.array([10.0, 10.0, 5.0, 5.0])
    #this can be asserted exactly, does not include a noise measurement
    assert np.allclose(ekf.state, expected, atol=1e-12)
 
 
def test_predict_grows_covariance():
    #process noise is always being added, it sould increase uncertainty
    #or, in wother words, the magnitude of the covariance matrix. 
    ekf = make_ekf(initial_covariance=np.eye(4))
    initial_trace = np.trace(ekf.covariance)
    for _ in range(10):
        ekf.predict()
    final_trace = np.trace(ekf.covariance)
    assert final_trace > initial_trace
 
 
def test_predict_preserves_covariance_symmetry():
    #P must remain symmetric after many predicts.
    ekf = make_ekf(initial_covariance=np.eye(4) * 50.0)
    for _ in range(100):
        ekf.predict()
    P = ekf.covariance
    #up to some uncertainty due to flooating point deviations
    assert np.allclose(P, P.T, atol=1e-9)
 

#check the jacobian 
def test_jacobian_matches_finite_differences():
    #the use of the jacobian in the EKF is based on the finite difference approximation
    #check if the jacobian matches the numerical approximation
    ekf = make_ekf(radar_position=Vec2(50.0, -30.0))
    #pick some state with a reasonable bearing and range
    x = np.array([200.0, 12.0, 150.0, -7.0])
    eps = 1e-6
    
    H_analytic = ekf._H(x)
    H_numeric = np.zeros((2, 4))
    for i in range(4):
        e = np.zeros(4)
        e[i] = eps
        z_plus = ekf._h(x + e)
        z_minus = ekf._h(x - e)
        H_numeric[:, i] = (z_plus - z_minus) / (2 * eps)
    
    #matches, up to some uncertainty, naturally
    assert np.allclose(H_analytic, H_numeric, atol=1e-6)
 
 
def test_jacobian_velocity_columns_are_zero():
    #this is just a standard test that would make sense if we were copmuting the jacobian everytime 
    #but we already have it expression hardcode
    #still good practice though 
    ekf = make_ekf()
    x = np.array([100.0, 5.0, 200.0, -3.0])
    H = ekf._H(x)
    # Column index 1 is vx, column index 3 is vy (per IVX=1, IVY=3 convention).
    assert np.allclose(H[:, 1], 0.0)
    assert np.allclose(H[:, 3], 0.0)
 
 

#updates tests
 
def test_update_reduces_covariance():
    # a reasonable measurement (acccording to our model) should reduce uncertanty on P
    #or, in other words, the magnitude of the covariance matrix.
    ekf = make_ekf(initial_covariance=np.eye(4) * 1000.0)
    trace_before = np.trace(ekf.covariance)
 
    #construct a reasonable measurement
    x = ekf.state
    range_m = np.hypot(x[0], x[2])
    bearing = np.arctan2(x[2], x[0])
    m = Measurement(t=0.0, range_m=range_m, bearing_rad=bearing, radar_id="r1")
    ekf.update(m)
    
    trace_after = np.trace(ekf.covariance)
    #check if the trace got smaller
    assert trace_after < trace_before
 
 
def test_update_preserves_covariance_symmetry():
    #check if the Joseph form is doing its job, by ensuring that P remains symmetric
    ekf = make_ekf()
    for _ in range(100):
        ekf.predict()
        x = ekf.state
        m = Measurement(
            t=0.0,
            range_m=np.hypot(x[0], x[2]) + np.random.normal(0, 10),
            bearing_rad=np.arctan2(x[2], x[0]) + np.random.normal(0, 0.01),
            radar_id="r1",
        )
        ekf.update(m)
    P = ekf.covariance
    #check simmetry
    assert np.allclose(P, P.T, atol=1e-9)
 
 
def test_update_bearing_wrap_short_path():
    # Place radar at origin and predict a target with bearing of ~-pi
    ekf = make_ekf(
        radar_position=Vec2(0.0, 0.0),
        initial_state=np.array([-1000.0, 0.0, 0.0, 0.0]),
        initial_covariance=np.diag([100.0, 100.0, 100.0, 100.0]),
    )
 
    state_before = ekf.state.copy()
    predicted_range = 1000.0
    #then change it to be ~pi. 
    #check if the cganhe on the bearing changes by 0.1 and not almost 2pi
    measured_bearing = -np.pi + 0.1
    m = Measurement(
        t=0.0,
        range_m=predicted_range,
        bearing_rad=measured_bearing,
        radar_id="r1",
    )
    ekf.update(m)
    #check how much it changed 
    state_change = np.linalg.norm(ekf.state - state_before)
    #if it was a difference of almost 2pi in the bearing, 
    #the resdual would compute 2pi + 0.1 which, after multplying by the Kalman gain, 
    #would increase significantly the value computed in for the state
    #so the difference in states would be too significant. 
    #this upper bound could be worked on
    assert state_change < 100.0
 
 
def test_filter_converges_on_clean_track():
    #check convergence of the filter
    #basically, check if it is doing what is supposed to do
    rng = np.random.default_rng(42)
    true_state = np.array([100.0, 10.0, 100.0, 5.0])
    radar_pos = Vec2(0.0, 0.0)
    dt = 1.0
    sigma_r = 5.0
    sigma_b = np.deg2rad(0.5)
 
    ekf = EKF(
        radar_position=radar_pos,
        dt=dt,
        q=0.1,
        sigma_r=sigma_r,
        sigma_b=sigma_b,
        initial_state=np.array([true_state[0], 0.0, true_state[2], 0.0]),
        initial_covariance=np.diag([sigma_r**2, 100.0, sigma_r**2, 100.0]),
    )
 
    F = np.array(
        [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float64
    )
 
    n_steps = 100
    for _ in range(n_steps):
        #propagate the true satte using our process model
        true_state = F @ true_state
        #create a measurement
        dx = true_state[0] - radar_pos.x
        dy = true_state[2] - radar_pos.y
        true_range = np.hypot(dx, dy)
        true_bearing = np.arctan2(dy, dx)
        #add some noise to the measurement
        m = Measurement(
            t=0.0,
            range_m=true_range + rng.normal(0, sigma_r),
            bearing_rad=true_bearing + rng.normal(0, sigma_b),
            radar_id="r1",
        )
        ekf.predict()
        ekf.update(m)
 
    #check if the estimate is close to the truth
    err = ekf.state - true_state
    pos_err = np.hypot(err[0], err[2])
    vel_err = np.hypot(err[1], err[3])
    #the upperbound are a bit empirical on what makes sense
    assert pos_err < 30.0
    assert vel_err < 5.0
 
 
def test_no_nan_or_inf_after_many_steps():
    #check if tthere are any Nan 
    rng = np.random.default_rng(0)
    ekf = make_ekf()
    for _ in range(100):
        ekf.predict()
        x = ekf.state
        m = Measurement(
            t=0.0,
            range_m=np.hypot(x[0], x[2]) + rng.normal(0, 10),
            bearing_rad=np.arctan2(x[2], x[0]) + rng.normal(0, 0.01),
            radar_id="r1",
        )
        ekf.update(m)

    assert np.all(np.isfinite(ekf.state))
    assert np.all(np.isfinite(ekf.covariance))