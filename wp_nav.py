import numpy as np
import threading 
import math

class WP_Navigator:
    def __init__(self, waypoints_px):
        self.waypoints = waypoints_px
        self.current_index = 0
        self.lock = threading.Lock()

    def get_next_waypoint(self):
        with self.lock:
            if self.current_index < len(self.waypoints):
                waypoint = self.waypoints[self.current_index]
                self.current_index += 1
                return waypoint
            else:
                return None
    
    # def diff_drive_controller(x, y, theta, target_x, target_y, linear_gain=0.5, angular_gain=2.0):
    #     dx = target_x - x
    #     dy = target_y - y

    #     target_angle = math.atan2(dy, dx)
    #     angle_diff = target_angle - theta
    #     angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

    #     distance = math.hypot(dx, dy)

    #     # Proportional control
    #     linear_velocity = linear_gain * distance
    #     angular_velocity = angular_gain * angle_diff

    #     return linear_velocity, angular_velocity


    def reset(self):
        with self.lock:
            self.current_index = 0