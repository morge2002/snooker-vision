# TODO: Implement a Kalman filter to estimate the state of the system. I haven't started this at all yet.
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def update(self, measurement):
        # prediction
        state_prediction = self.state
        covariance_prediction = self.covariance + self.process_noise

        # update
        kalman_gain = covariance_prediction / (covariance_prediction + self.measurement_noise)
        self.state = state_prediction + kalman_gain * (measurement - state_prediction)
        self.covariance = (1 - kalman_gain) * covariance_prediction

        return self.state
