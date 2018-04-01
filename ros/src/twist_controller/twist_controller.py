import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED = 0.1  # Requires a positive number so that the Yaw Controller doesn't have a division by zero error.

# PID for throttle variables.  Initial values of 0.3, 0.1, 0., 0., 0.2 from walkthrough and were determined
# experimentally TODO: Try different values
KP = 0.3
KI = 0.1
KD = 0.
MN = 0.  # Minimum Throttle Value
MX = 0.8  # Maximum Throttle Value

# Low Pass Filter Variables.  Initial values of 0.5, 0.02 per walkthrough
# TODO: Try different values
TAU = 0.5  # 1 / (2pi*tau) = cutoff frequency
TS = 0.02  # Sample Time


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.yaw_controller = YawController(
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            min_speed=MIN_SPEED,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle
        )

        self.throttle_controller = PID(KP, KI, KD, MN, MX)
        # The current velocity coming in from the messages is noisy, this smooths it out
        self.velocity_lowpassfilter = LowPassFilter(TAU, TS)

        self.last_time = rospy.get_time()

    def control(self, target_linear_velocity, target_angular_velocity, current_linear_velocity, dbw_enabled, *args,
                **kwargs):

        if not dbw_enabled:
            # Manual driving, reset the PID.
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_linear_velocity = self.velocity_lowpassfilter.filt(current_linear_velocity)

        steer = self.yaw_controller.get_steering(current_linear_velocity, target_angular_velocity, target_linear_velocity)

        velocity_diff = target_linear_velocity - current_linear_velocity

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(velocity_diff, sample_time)
        brake = 0.0

        if target_linear_velocity == 0. and current_linear_velocity < 0.1:
            throttle = 0
            brake = 400 # 400 N*m holds the car stopped at a light

        elif throttle < 0.1 and velocity_diff < 0:
            throttle = 0
            decel = max(velocity_diff, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # To calculate torque in N*m

        return throttle, brake, steer
