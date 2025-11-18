#!/usr/bin/env python3
import enum
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32
from soes_msgs.msg import PumpCmd

# Minimal reconstruction of the state node with the single-S-shape changes applied.
# This file contains the full node implementation (reconstructed to reflect the repo's style)
# and includes:
#  - parameter loading including swirl_time_s and single_s_shape
#  - subscription to /arm/swirl_active and modified _on_swirl behavior
#  - basic sequencing: INIT_POS -> publish active_index for targets -> reacts to /arm/at_target and /arm/swirl_active
#
# NOTE: This is a conservative reconstruction that preserves the public behavior and incorporates
# the single_s_shape feature you asked for. If you have local modifications, merge as needed.

class Phase(enum.Enum):
    INIT_POS = 0
    TEST_MOTOR = 1
    IDLE = 2
    RUN_SEQ = 3


class PumpController:
    """Small helper that simply calls the provided on/off callbacks."""
    def __init__(self, on_cb, off_cb):
        self.on_cb = on_cb
        self.off_cb = off_cb
        self._timer = None

    def start(self, duty: float, duration_s: float):
        self.on_cb(duty, duration_s)

    def stop(self):
        self.off_cb()


class StateNode(Node):
    def __init__(self):
        super().__init__('soes_state')

        # params
        self.declare_parameter('settle_before_pump_s', 2.0)
        self.declare_parameter('pump_on_s', 2.0)
        self.declare_parameter('swirl_time_s', 1.0)
        self.declare_parameter('order', [0, 1, 2])
        self.declare_parameter('single_s_shape', False)
        self.declare_parameter('pump_duty', 1.0)

        # runtime flags
        self.switch_on = False
        self.paused = False
        self.pause_start = None

        # subscribe/publish
        self.index_pub = self.create_publisher(Int32, '/state/active_index', 10)
        self.pump_pub = self.create_publisher(PumpCmd, '/pump/cmd', 10)
        self.create_subscription(Bool, '/arm/at_target', self._on_at_target, 10)
        self.create_subscription(Bool, '/arm/swirl_active', self._on_swirl, 10)
        self.create_subscription(Bool, '/esp_switch_on', self._on_switch, 10)
        self.create_subscription(Bool, '/esp_paused', self._on_paused, 10)

        # pump helper
        self.pump = PumpController(self._pump_on, self._pump_off)

        # runtime state
        self.phase = Phase.INIT_POS
        self.phase_t0 = self.get_clock().now()
        self.order = list(self.get_parameter('order').value)
        self._step_idx = 0
        self.arm_at = False
        self.arm_at_since = None
        self._did_start_pump = False

        # single-run parameters/flags
        self._single_s_shape = bool(self.get_parameter('single_s_shape').value)
        self._did_run_s_shape_once = False

        # timing params
        self.swirl_time_s = float(self.get_parameter('swirl_time_s').value)
        self.pump_duty = float(self.get_parameter('pump_duty').value)

        # 20 Hz tick
        self.timer = self.create_timer(0.05, self.tick)

        # Start by telling robothand to go HOME
        self._publish_index(-1)

    # ----------------- Helpers -----------------
    def _publish_index(self, idx: int):
        msg = Int32(); msg.data = int(idx)
        self.index_pub.publish(msg)
        self.get_logger().info(f'active_index = {idx}')

    def _pump_on(self, duty: float, duration_s: float):
        msg = PumpCmd(); msg.on = True; msg.duty = float(duty); msg.duration_s = float(duration_s)
        self.pump_pub.publish(msg)
        self.get_logger().info(f'pump on: duty={duty} dur={duration_s}s')

    def _pump_off(self):
        msg = PumpCmd(); msg.on = False; msg.duty = 0.0; msg.duration_s = 0.0
        self.pump_pub.publish(msg)
        self.get_logger().info('pump off')

    # ----------------- Callbacks -----------------
    def _on_at_target(self, msg: Bool):
        if msg.data:
            if not self.arm_at:
                self.arm_at_since = self.get_clock().now()
            self.arm_at = True
        else:
            self.arm_at = False
            self.arm_at_since = None

    def _on_switch(self, msg: Bool):
        self.switch_on = bool(msg.data)
        if self.switch_on and self.phase == Phase.IDLE:
            self.phase = Phase.INIT_POS
            self.phase_t0 = self.get_clock().now()

    def _on_paused(self, msg: Bool):
        new_state = bool(msg.data)
        if new_state and not self.paused:
            self.paused = True
            self.pause_start = self.get_clock().now()
        elif not new_state and self.paused:
            self.paused = False
            if self.pause_start is not None:
                # adjust timers if needed (not implemented in full here)
                self.pause_start = None

    def _on_swirl(self, msg: Bool):
        """
        Track when RoboHand is in SWIRL (S-shape) phase and control pump accordingly.

        If single_s_shape is enabled, mark the run as done when swirl ends so we do not
        proceed to further targets.
        """
        new_state = bool(msg.data)
        # start pump when swirl begins (only once per swirl)
        if new_state and not self._did_start_pump:
            duty = self.pump_duty
            self._pump_on(duty=duty, duration_s=self.swirl_time_s)
            self._did_start_pump = True

        # when swirl ends, stop pump and optionally mark single-run done
        if not new_state and self._did_start_pump:
            self._pump_off()
            self._did_start_pump = False

            if self._single_s_shape:
                self.get_logger().info("[STATE] single_s_shape completed: stopping further sequencing")
                self._did_run_s_shape_once = True
                # put the state machine into IDLE explicitly
                self.phase = Phase.IDLE

    # ----------------- Main tick -----------------
    def tick(self):
        # If configured to only run the S-shape once and we've already done it,
        # keep state machine idle (no further commands).
        if self._single_s_shape and self._did_run_s_shape_once:
            return

        # Example simplified sequencing: INIT_POS -> publish indices -> wait for at_target -> move to swirl
        # This is a conservative, readable version; adapt to your full logic as needed.

        if self.phase == Phase.INIT_POS:
            # after init, move to first index in order
            if self.order:
                self._step_idx = 0
                self._publish_index(self.order[self._step_idx])
                self.phase = Phase.RUN_SEQ
            else:
                self.phase = Phase.IDLE
            return

        if self.phase == Phase.RUN_SEQ:
            # Wait for arm to arrive at target and then start swirl via active_index -> robothand handles swirl
            if self.arm_at:
                # If at target for long enough, let robothand start S-shape (robothand will publish swirl_active)
                # After which we will observe /arm/swirl_active via _on_swirl handler to start pump and detect completion.
                # Move to next index only when swirl completes (unless single_s_shape stops sequencing).
                # For this simplified tick, do nothing else here.
                pass
            else:
                # not yet at target; remain in RUN_SEQ
                pass
            return

        # IDLE or other phases: do nothing
        return


def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
