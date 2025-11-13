import enum
import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32
from soes_msgs.msg import JointTargets, CupcakeCenters


# State Machine Phases
class Phase(enum.Enum):
    WAIT  = 0
    MOVE  = 1
    SWIRL = 2


# Main Node
class RoboHandNode(Node):
    def __init__(self):
        super().__init__('soes_robothand')

        # Control Parameters:
        self.declare_parameter('rate_hz', 50.0)  # Control Update Rate
        self.declare_parameter('pos_tol_m', 0.003)  # Position Tolerance
        self.declare_parameter('settle_s', 0.20)  # Settling time before switching phases

        # Robot link lengths
        self.declare_parameter('link_lengths_m', [0.00, 0.14, 0.12, 0.04])     # [L1..L4]

        # IK tuning
        self.declare_parameter('kp_cart', 3.0)
        self.declare_parameter('damping_lambda', 0.1)
        self.declare_parameter('joint_limits', {
            'qdot': [1.5, 1.5, 1.5, 1.5],
            'qmin': [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2],
            'qmax': [ math.pi,  math.pi/2,  math.pi/2,  math.pi/2],
        })

        # Spiral parameters
        self.declare_parameter('R0', 0.025)
        self.declare_parameter('turns', 3)
        self.declare_parameter('alpha', -0.03)
        self.declare_parameter('height', 0.04)
        self.declare_parameter('omega', 0.5)  # Angular Speed [rad/s] *Change if robot can't keep up*

        # Load parameters
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.dt = 1.0 / self.rate_hz
        self.pos_tol = float(self.get_parameter('pos_tol_m').value)
        self.settle_s = float(self.get_parameter('settle_s').value)

        self.links = np.array(self.get_parameter('link_lengths_m').value, dtype=float)
        self.L1, self.L2, self.L3, self.L4 = map(float, self.links)

        self.kp = float(self.get_parameter('kp_cart').value)
        self.lmbda = float(self.get_parameter('damping_lambda').value)

        limits = self.get_parameter('joint_limits').value
        self.qdot_lim = np.array(limits['qdot'], dtype=float)
        self.q_min = np.array(limits['qmin'], dtype=float)
        self.q_max = np.array(limits['qmax'], dtype=float)

        self.R0 = float(self.get_parameter('R0').value)
        self.turns = int(self.get_parameter('turns').value)
        self.alpha = float(self.get_parameter('alpha').value)
        self.height = float(self.get_parameter('height').value)
        self.omega = float(self.get_parameter('omega').value)
        self.theta_max = 2 * math.pi * self.turns
        self.s = self.height / self.theta_max

        # ROS I/O
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_sub   = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.center_sub  = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)

        # Runtime
        self.q = np.zeros(4, dtype=float)
        self.active_index: int = -1
        self.centers: Optional[list[Tuple[float,float,float]]] = None

        self.phase = Phase.WAIT
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz: Optional[np.ndarray] = None

        # Spiral state
        self.spiral_theta = 0.0
        self.spiral_center: Optional[np.ndarray] = None

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('soes_robothand (spiral DLS IK) initialized.')

    # Subscriptions
    def _on_index(self, msg: Int32):
        prev = self.active_index
        self.active_index = int(msg.data)
        self.get_logger().info(f"active_index: {prev} -> {self.active_index}")
        self._on_index_changed()

    def _on_centers(self, msg: CupcakeCenters):
        # assume msg.centers is an iterable of point-like messages with x,y,z
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]
        if len(self.centers) < 3:
            self.get_logger().warn("centers < 3; waiting for all three targets")
            return
        if self.phase == Phase.WAIT and self.active_index in (0,1,2):
            self._start_move_to_current_index()

    # Phase Control/Target Logic
    def _on_index_changed(self):
        if self.active_index in (0,1,2) and self.centers and len(self.centers) >= 3:
            self._start_move_to_current_index()
        else:
            self._enter(Phase.WAIT, None)

    def _start_move_to_current_index(self):
        # protect against invalid index
        if self.centers is None or not (0 <= self.active_index < len(self.centers)):
            return
        xyz = self.centers[self.active_index]
        self._enter(Phase.MOVE, xyz)

    def _start_swirl_at_current_index(self):
        # Set swirl center and reset spiral progression
        if self.centers is None or not (0 <= self.active_index < len(self.centers)):
            return
        self.spiral_center = np.array(self.centers[self.active_index], dtype=float)
        self.spiral_theta = 0.0
        # start SWIRL; des_xyz will be computed each tick from spiral_center
        self._enter(Phase.SWIRL, self.spiral_center.copy())

    def _enter(self, new_phase: Phase, xyz: Optional[Tuple[float,float,float]]):
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz = np.array(xyz, dtype=float) if xyz is not None else None
        self.get_logger().info(f"[ROBOHAND] -> {self.phase.name}  des={self.des_xyz}")

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

    # Kinematics (Analytic FK & Jacobian)
    def fk_xyz(self, q: np.ndarray) -> np.ndarray:
        q1, q2, q3, q4 = q
        L1, L2, L3, L4 = self.L1, self.L2, self.L3, self.L4
        r_fk = L2*math.cos(q2) + L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        z_fk = L1 + L2*math.sin(q2) + L3*math.sin(q2+q3) + L4*math.sin(q2+q3+q4)
        x_fk = r_fk * math.cos(q1)
        y_fk = r_fk * math.sin(q1)
        return np.array([x_fk, y_fk, z_fk], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q1, q2, q3, q4 = q
        L1, L2, L3, L4 = self.L1, self.L2, self.L3, self.L4

        r_fk = L2*math.cos(q2) + L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dr_dq2 = -L2*math.sin(q2) - L3*math.sin(q2+q3) - L4*math.sin(q2+q3+q4)
        dr_dq3 = -L3*math.sin(q2+q3) - L4*math.sin(q2+q3+q4)
        dr_dq4 = -L4*math.sin(q2+q3+q4)
        dz_dq2 =  L2*math.cos(q2) + L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dz_dq3 =  L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dz_dq4 =  L4*math.cos(q2+q3+q4)

        J = np.zeros((3,4))
        J[:,0] = [-r_fk*math.sin(q1), r_fk*math.cos(q1), 0.0]
        J[:,1] = [math.cos(q1)*dr_dq2, math.sin(q1)*dr_dq2, dz_dq2]
        J[:,2] = [math.cos(q1)*dr_dq3, math.sin(q1)*dr_dq3, dz_dq3]
        J[:,3] = [math.cos(q1)*dr_dq4, math.sin(q1)*dr_dq4, dz_dq4]
        return J

    # IK Step (Resolved-rate with DLS)
    def ik_step(self, des_xyz: np.ndarray, xdot_ff: Optional[np.ndarray] = None):
        cur_xyz = self.fk_xyz(self.q)
        err = des_xyz - cur_xyz
        err_norm = float(np.linalg.norm(err))

        # settle logic
        if err_norm <= self.pos_tol:
            if self.last_within_tol is None:
                self.last_within_tol = self.get_clock().now()
        else:
            self.last_within_tol = None

        # desired cartesian velocity: PD + optional feedforward
        v = self.kp * err
        if xdot_ff is not None:
            v = v + xdot_ff

        J = self.jacobian(self.q)
        JJt = J @ J.T
        
        # Damped least squares solution
        qdot = J.T @ np.linalg.solve(JJt + (self.lmbda ** 2) * np.eye(3), v)
        qdot = np.clip(qdot, -self.qdot_lim, self.qdot_lim)
        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)

        msg = JointTargets()
        msg.position = [float(a) for a in self.q]
        msg.velocity = [float(w) for w in qdot]
        msg.use_velocity = True
        self.targets_pub.publish(msg)

    # Main Loop
    def _tick(self):
        if self.centers is None:
            return

        if self.phase == Phase.MOVE and self.des_xyz is not None:
            self.ik_step(self.des_xyz)

            within_settle = (
                self.last_within_tol is not None and
                (self.get_clock().now() - self.last_within_tol) >= Duration(seconds=self.settle_s)
            )
            if within_settle:
                self._start_swirl_at_current_index()

        elif self.phase == Phase.SWIRL and self.spiral_center is not None:
            # Generate spiral offset relative to center
            if self.spiral_theta <= self.theta_max:
                r = self.R0 * (1 + self.alpha * self.spiral_theta)
                dx = r * math.cos(self.spiral_theta)
                dy = r * math.sin(self.spiral_theta)
                dz = self.s * self.spiral_theta
                swirl_xyz = self.spiral_center + np.array([dx, dy, dz])
                # feedforward velocity
                rdot = self.R0 * self.alpha * self.omega
                xdot = rdot * math.cos(self.spiral_theta) - r * self.omega * math.sin(self.spiral_theta)
                ydot = rdot * math.sin(self.spiral_theta) + r * self.omega * math.cos(self.spiral_theta)
                zdot = self.s * self.omega
                xdot_ff = np.array([xdot, ydot, zdot])
                self.ik_step(swirl_xyz, xdot_ff)
                self.spiral_theta += self.omega * self.dt
            else:
                self._enter(Phase.WAIT, None)


def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
