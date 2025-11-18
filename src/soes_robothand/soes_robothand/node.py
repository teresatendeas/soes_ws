#!/usr/bin/env python3
import enum
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Bool, Int32


class Phase(enum.Enum):
    IDLE = 0
    MOVE_TO_INDEX = 1
    SETTLE = 2
    SWIRL = 3
    # The original node supported multi-step sequences and more phases
    # (e.g. sequence of indexes, camera, roll tray, etc.). For the single-step
    # mode requested (INIT_POS -> STEP0 -> DONE), those phases are preserved
    # below but are not used by the "single-step" default flow.
    UNUSED_STEP1 = 4  # retained but unused in normal single-step flow
    UNUSED_STEP2 = 5  # retained but unused in normal single-step flow


class RoboHandNode(Node):
    """
    Simplified RoboHand node adapted to the single-step mode:
      - It listens to /state/active_index (Int32)
        * -1 -> command robot to go HOME (publish /arm/at_target True after settle)
        * >=0 -> treat any non-negative index as "STEP0" in the single-step flow:
                 move to index, wait settle_before_pump_s, then assert /arm/swirl_active True
                 for swirl_time_s and then assert False (which signals STEP0 complete to state node)
      - Other multi-step behaviors remain in the codebase but are commented/marked as unused.
    This mirrors the change to the state machine that stops at STEP0 (phase 1).
    """
    def __init__(self):
        super().__init__('soes_robothand')

        # Parameters (kept for compatibility; some are UNUSED in single-step flow)
        self.declare_parameter('settle_before_pump_s', 0.6)
        self.declare_parameter('swirl_time_s', 1.0)
        self.declare_parameter('order', [0, 1, 2])  # retained but only index 0 is used by state machine now

        self.t_settle = float(self.get_parameter('settle_before_pump_s').value)
        self.t_swirl = float(self.get_parameter('swirl_time_s').value)
        self.order = list(self.get_parameter('order').value)

        # Subscribers / Publishers
        self.active_index_sub = self.create_subscription(Int32, '/state/active_index', self.on_active_index, 10)
        # Publish whether the arm is at its commanded target (used by state node to gate transitions)
        self.at_target_pub = self.create_publisher(Bool, '/arm/at_target', 10)
        # Publish swirl active flag used by state node to control the pump
        self.swirl_pub = self.create_publisher(Bool, '/arm/swirl_active', 10)

        # Runtime
        self.phase = Phase.IDLE
        self.phase_t0 = self.get_clock().now()
        self.pending_index: Optional[int] = None  # currently commanded index from /state/active_index
        self._timer = self.create_timer(0.05, self.tick)

        self.get_logger().info('soes_robothand: ready (single-step adapted).')

        # Initially, announce that we're at HOME (so state node can progress to START)
        # Note: state node expects /arm/at_target to be True and stable for t_settle to start STEP0.
        # Publish False briefly then True after small delay to simulate startup settling.
        self._publish_at_target(False)
        self.create_timer(0.2, lambda: self._publish_at_target(True))

    # ---------- helpers ----------
    def _enter(self, new_phase: Phase):
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self.get_logger().info(f'[ROBOHAND] -> {self.phase.name}')

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

    def _publish_at_target(self, state: bool):
        msg = Bool(); msg.data = bool(state)
        self.at_target_pub.publish(msg)
        self.get_logger().debug(f'/arm/at_target = {state}')

    def _publish_swirl(self, state: bool):
        msg = Bool(); msg.data = bool(state)
        self.swirl_pub.publish(msg)
        self.get_logger().info(f'/arm/swirl_active = {state}')

    # ---------- callbacks ----------
    def on_active_index(self, msg: Int32):
        """
        Handle incoming active_index from state node.

        For single-step mode:
          - -1 -> go to HOME (publish at_target after settle)
          - >=0 -> treat as STEP0 (move to index, settle, then run SWIRL for t_swirl)
        Any indices >0 (STEP1/STEP2) are acknowledged but ignored because the state
        machine was changed to stop at phase 1 (STEP0). The logic below intentionally
        logs that those are unsupported in the single-step flow.
        """
        idx = int(msg.data)
        self.get_logger().info(f'Received active_index = {idx}')
        # Save requested index and let tick() handle transitions
        self.pending_index = idx

    # ---------- main tick ----------
    def tick(self):
        # No global pause handling here; robothand simply responds to commands.
        # Process pending command if present
        if self.pending_index is not None:
            idx = self.pending_index
            self.pending_index = None

            if idx == -1:
                # Command: go HOME
                # Simulate moving to HOME: during move, publish at_target False,
                # after settle time publish at_target True.
                self.get_logger().info('Command: HOME (go to INIT_POS).')
                self._enter(Phase.MOVE_TO_INDEX)
                self._publish_at_target(False)
                # Transition to SETTLE, track time via phase and tick() loop
                self._enter(Phase.SETTLE)

            elif idx >= 0:
                # Command: move to an index. In single-step flow we will treat any
                # non-negative index as STEP0 (single active step), then run SWIRL and
                # report completion by publishing /arm/swirl_active False after swirl.
                # If the index is greater than 0 we still accept it but log that
                # multi-step indexing is currently disabled on the state machine side.
                if idx != 0:
                    self.get_logger().warn('Received index >=1 — multi-step flow is disabled; treating as STEP0.')
                else:
                    self.get_logger().info('Command: STEP0 (move to target index).')

                self._enter(Phase.MOVE_TO_INDEX)
                # Simulate movement completion immediately for this node (in real hardware you'd wait)
                # Indicate not at target during movement
                self._publish_at_target(False)
                # After MOVE we go to SETTLE to allow state node to see arm_at_since and begin its logic.
                self._enter(Phase.SETTLE)

            else:
                # Unexpected index; ignore but log.
                self.get_logger().warn(f'Unexpected active_index {idx}; ignoring.')

        # Phase-based state machine
        if self.phase == Phase.IDLE:
            return

        elif self.phase == Phase.MOVE_TO_INDEX:
            # In this simplified node we immediately proceed to SETTLE (movement simulated above),
            # so MOVE_TO_INDEX is typically short-lived. If you have real movement feedback, you can
            # replace this with position tracking and only switch to SETTLE after the hardware reports arrival.
            pass  # movement was simulated in on_active_index

        elif self.phase == Phase.SETTLE:
            # Wait settle_before_pump_s then mark at_target True and start SWIRL
            if self._elapsed() >= self.t_settle:
                self.get_logger().info('Settle complete; announcing at_target and starting SWIRL.')
                self._publish_at_target(True)
                # Start swirling phase (this will be seen by the state node which controls the pump)
                self._enter(Phase.SWIRL)
                # Immediately assert swirl_active True
                self._publish_swirl(True)

        elif self.phase == Phase.SWIRL:
            # Keep swirl_active True for configured time, then turn it off.
            if self._elapsed() >= self.t_swirl:
                self.get_logger().info('Swirl complete; de-asserting swirl_active.')
                self._publish_swirl(False)
                # In the previous multi-step flow we would then either wait for the next index
                # or transition to further steps (STEP1/STEP2/CAMERA). For the "stop at phase 1"
                # single-step mode, we simply go IDLE and let the state node move to IDLE as well.
                self._enter(Phase.IDLE)

        # The following phases / behavior were part of a larger multi-step sequence previously.
        # They are intentionally kept (commented/explained) for future re-enabling, but they are
        # not used when the state machine is configured to stop after STEP0.
        # Examples of previously supported extensions (kept for reference):
        #
        # - Handling multiple ordered indices from 'order' parameter
        # - Running camera quality checks and acknowledging them
        # - Triggering tray roll via a service call
        #
        # All such code should be re-integrated here if the state machine is later restored
        # to support full multi-step cycles.

def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
