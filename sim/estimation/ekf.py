import numpy as np
from sim.core import Vec2, Measurement, TargetStateView
from sim.core.types import IX, IY, IVX, IVY, STATE_DIM_CV


class EKF:

    def __init__(self, radar_position: Vec2, dt: float, q: float, sigma_r: float, sigma_b: float,
                 initial_state: np.ndarray, initial_covariance: np.ndarray) -> None:
        #radar position (fixed)
        self.radar_position = radar_position
        #time step in seconds
        self.dt = dt
        #spectral density for the process noise Q
        self.q = q
        #noise factor for the range measurement noise
        self.sigma_r = sigma_r
        #noise factor for the bearing measurement noise
        self.sigma_b = sigma_b
        #initial state vector shape = (STATE_DIM_CV, )
        self.x = initial_state
        #initial state covariance matrix shape = (STATE_DIM_CV, STATE_DIM_CV)
        self.P = initial_covariance

        #process model matrix
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]], dtype=np.float64)
        
        #process noise covariance matrix
        self.Q = self.q * np.array([[dt**4/4, dt**3/2, 0, 0],
                                    [dt**3/2, dt**2, 0, 0],
                                    [0, 0, dt**4/4, dt**3/2],
                                    [0, 0, dt**3/2, dt**2]], dtype=np.float64)
        
        #measurement noise covariance matrix
        self.R = np.diag([self.sigma_r**2, self.sigma_b**2])

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()
    
    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()
    
    @property
    def state_view(self) -> TargetStateView:
        return TargetStateView.from_array(self.x)


    def predict(self) -> None:

        #predict the next state using the process model
        self.x = self.F @ self.x
        #predict the next state covariance
        self.P = self.F @ self.P @ self.F.T + self.Q


    def _h(self, x: np.ndarray) -> np.ndarray:
        #compute the expected measurement given the state x
        dx = x[IX] - self.radar_position.x
        dy = x[IY] - self.radar_position.y
        range_m = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx)
        return np.array([range_m, bearing])
    

    def _H(self, x: np.ndarray) -> np.ndarray:
        #compute the jacobian of the measurement function h at the state x
        dx = x[IX] - self.radar_position.x
        dy = x[IY] - self.radar_position.y
        r = np.sqrt(dx**2 + dy**2)
        r2 = r**2
        return np.array([
            [ dx/r,  0,  dy/r,  0],
            [-dy/r2, 0,  dx/r2, 0],
        ])


    def update(self, measurement: Measurement) -> None:
        #compute the expected measurement given the predicted state
        x_pred = self.x
        P_pred = self.P

        #apply the measurement function
        z_pred = self._h(x_pred)
        #turn the measurement into a vector
        z = np.array([measurement.range_m, measurement.bearing_rad])

        #compute the measurement residual
        y = z - z_pred
        #just to make sure the bearing residual is in the interval [-pi, pi]
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

        #compute the jacobian matrix at the predicted state
        H = self._H(x_pred)
        #compute S and K (this are the std KF equations)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        #compute estimates and update state and covariance
        self.x = x_pred + K @ y
        #I am using the Joseph form to ensure numerical stability and positive semi-definiteness of 
        # the covariance matrix
        I_KH = np.eye(STATE_DIM_CV) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

