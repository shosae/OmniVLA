#!/usr/bin/env python3
import math
import select
import sys
import termios
import tty

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from pop.driving_base import DrivingBase
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_srvs.srv import SetBool


class OmniDriveController(Node):
    """Subscribe to /cmd_vel and drive the omni base."""

    def __init__(self) -> None:
        super().__init__("omni_drive_controller")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("can_baudrate", 500000)
        self.declare_parameter("wheel_radius_m", 0.041)
        self.declare_parameter("wheel_to_center_m", 0.135)
        self.declare_parameter("wheel_angles_deg", [60.0, 300.0, 180.0])
        self.declare_parameter("drive_enabled", True)

        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.can_interface = self.get_parameter("can_interface").value
        self.can_baudrate = self.get_parameter("can_baudrate").value
        self.wheel_radius = self.get_parameter("wheel_radius_m").value
        self.wheel_to_center = self.get_parameter("wheel_to_center_m").value
        self.wheel_angles_deg = self.get_parameter("wheel_angles_deg").value
        self.drive_enabled = bool(self.get_parameter("drive_enabled").value)
        self.stopped_for_disable = False
        self.stdin_settings = None
        self.keyboard_enabled = sys.stdin.isatty()

        angles_rad = [math.radians(deg) for deg in self.wheel_angles_deg]
        self.sin = [math.sin(rad) for rad in angles_rad]
        self.cos = [math.cos(rad) for rad in angles_rad]

        self._init_velocity_mapping()

        self.driver = DrivingBase(self.can_interface, self.can_baudrate)
        self.subscription = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )
        self.param_callback_handle = self.add_on_set_parameters_callback(
            self.on_set_parameters
        )
        self.enable_service = self.create_service(
            SetBool,
            "~/set_enabled",
            self.handle_set_enabled,
        )
        self.key_timer = None

        if self.keyboard_enabled:
            self.stdin_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.key_timer = self.create_timer(0.05, self.poll_keyboard)

        self.get_logger().info(
            f"Listening on {self.cmd_vel_topic}, CAN={self.can_interface}@{self.can_baudrate}, "
            f"drive_enabled={self.drive_enabled}"
        )
        if self.keyboard_enabled:
            self.get_logger().info("Press SPACE in this terminal to toggle drive enable.")
        else:
            self.get_logger().warn("stdin is not a TTY; keyboard space toggle is disabled.")

    def _init_velocity_mapping(self) -> None:
        self.cmd_values = list(range(81))
        self.actual_velocities = [
            0, 2.00832, 2.11070, 2.21423, 2.31454, 2.41930, 2.52994, 2.63416, 2.73842,
            2.84276, 2.94427, 3.04667, 3.14937, 3.24979, 3.34983, 3.45665, 3.55730,
            3.65918, 3.76559, 3.86499, 3.96927, 4.07428, 4.17111, 4.27926, 4.38213,
            4.48546, 4.59145, 4.69252, 4.79835, 4.90096, 5.00738, 5.11186, 5.21490,
            5.32083, 5.42681, 5.53944, 5.63574, 5.74270, 5.84779, 5.95286, 6.05930,
            6.16499, 6.26801, 6.37679, 6.47986, 6.58630, 6.69247, 6.79808, 6.90474,
            7.00993, 7.11286, 7.21733, 7.32155, 7.42807, 7.53386, 7.63807, 7.74297,
            7.81623, 7.92091, 8.02478, 8.12866, 8.25195, 8.32442, 8.41934, 8.57043,
            8.67347, 8.77555, 8.91054, 9.01008, 9.11378, 9.25121, 9.35359, 9.45801,
            9.55409, 9.65882, 9.76252, 9.85996, 9.96249, 10.06160, 10.16744, 10.27037,
        ]
        self.max_wheel_velocity = max(self.actual_velocities)

    def vel_to_cmd(self, target_velocity: float) -> int:
        abs_velocity = abs(target_velocity)
        sign = 1 if target_velocity >= 0.0 else -1

        if abs_velocity > self.max_wheel_velocity:
            return sign * 80

        cmd = np.interp(abs_velocity, self.actual_velocities, self.cmd_values)
        return sign * int(round(cmd))

    def stop_robot(self) -> None:
        self.driver.stop()

    def set_drive_enabled(self, enabled: bool) -> None:
        if self.drive_enabled == enabled:
            return

        self.drive_enabled = enabled
        if not self.drive_enabled:
            self.stop_robot()
            self.stopped_for_disable = True
        else:
            self.stopped_for_disable = False
        self.get_logger().info(f"drive_enabled set to {self.drive_enabled}")

    def on_set_parameters(self, params):
        for param in params:
            if param.name != "drive_enabled":
                continue
            self.set_drive_enabled(bool(param.value))
        return SetParametersResult(successful=True)

    def handle_set_enabled(self, request, response):
        self.set_drive_enabled(bool(request.data))
        response.success = True
        response.message = f"drive_enabled={self.drive_enabled}"
        return response

    def read_key(self) -> str:
        if not self.keyboard_enabled or self.stdin_settings is None:
            return ""

        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ""
        return key

    def poll_keyboard(self) -> None:
        key = self.read_key()
        if key == " ":
            self.set_drive_enabled(not self.drive_enabled)
        elif key == "\x03":
            self.get_logger().info("Ctrl-C received from keyboard input.")
            rclpy.shutdown()

    def cmd_vel_callback(self, msg: Twist) -> None:
        if not self.drive_enabled:
            if not self.stopped_for_disable:
                self.stop_robot()
                self.stopped_for_disable = True
            return

        self.stopped_for_disable = False

        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        v1 = -self.sin[0] * vx + self.cos[0] * vy + self.wheel_to_center * wz
        v2 = -self.sin[1] * vx + self.cos[1] * vy + self.wheel_to_center * wz
        v3 = -self.sin[2] * vx + self.cos[2] * vy + self.wheel_to_center * wz

        w1 = v1 / self.wheel_radius
        w2 = v2 / self.wheel_radius
        w3 = v3 / self.wheel_radius

        max_wheel = max(abs(w1), abs(w2), abs(w3))
        if max_wheel > self.max_wheel_velocity:
            scale = self.max_wheel_velocity / max_wheel
            w1 *= scale
            w2 *= scale
            w3 *= scale

        cmd1 = self.vel_to_cmd(w1)
        cmd2 = self.vel_to_cmd(w2)
        cmd3 = self.vel_to_cmd(w3)

        self.driver.wheel_vec[0] = DrivingBase.WHEEL_CENTER + cmd1
        self.driver.wheel_vec[1] = DrivingBase.WHEEL_CENTER + cmd2
        self.driver.wheel_vec[2] = DrivingBase.WHEEL_CENTER + cmd3
        self.driver.transfer()

    def destroy_node(self) -> bool:
        self.get_logger().info("Stopping the robot...")
        self.stop_robot()
        if self.keyboard_enabled and self.stdin_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.stdin_settings)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OmniDriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
