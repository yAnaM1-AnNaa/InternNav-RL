"""Adapt InternNav velocity commands to the AgileX Bunker base driver."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


class VelocityBridge(Node):
    def __init__(self):
        super().__init__("bunker_velocity_bridge")
        self.max_linear = self.positive_parameter("max_linear", 0.3)
        self.max_angular = self.positive_parameter("max_angular", 0.4)
        self.command_timeout = self.positive_parameter("command_timeout", 0.5)
        self.last_command = None
        self.output = self.create_publisher(Twist, "/internnav/bunker/cmd_vel", 1)
        self.input = self.create_subscription(Twist, "/cmd_vel_bridge", self.on_command, 1)
        self.watchdog = self.create_timer(
            min(0.05, self.command_timeout), self.stop_if_stale, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    def positive_parameter(self, name, default):
        value = self.declare_parameter(name, default).value
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def on_command(self, message):
        linear, angular = message.linear.x, message.angular.z
        if not (math.isfinite(linear) and math.isfinite(angular)):
            self.stop()
            self.get_logger().warning("Ignored a non-finite velocity command", throttle_duration_sec=2.0)
            return

        command = Twist()
        command.linear.x = max(-self.max_linear, min(self.max_linear, linear))
        command.angular.z = max(-self.max_angular, min(self.max_angular, angular))
        self.output.publish(command)
        self.last_command = time.monotonic()

    def stop_if_stale(self):
        if self.last_command is not None and time.monotonic() - self.last_command >= self.command_timeout:
            self.stop()

    def stop(self):
        self.last_command = None
        self.output.publish(Twist())


def main():
    # Keep the context alive on Ctrl+C long enough to request zero velocity.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = VelocityBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                if rclpy.ok():
                    node.stop()
                    node.output.wait_for_all_acked(Duration(seconds=0.2))
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
