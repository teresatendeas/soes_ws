#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from soes_msgs.msg import JointTargets
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

class GraphNode(Node):
    def __init__(self):
        super().__init__('soes_graph')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.sub = self.create_subscription(JointTargets, '/arm/joint_targets', self.on_joint, qos)

        # history length (seconds)
        self.window_s = 10.0
        self.rate_hz = 50.0
        self.maxlen = int(self.window_s * self.rate_hz)

        self.time_data = deque(maxlen=self.maxlen)
        self.data = [deque(maxlen=self.maxlen) for _ in range(4)]
        self.use_velocity = False

        # matplotlib setup
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        self.lines = []
        labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']
        for i, ax in enumerate(self.axs):
            line, = ax.plot([], [], lw=2)
            ax.set_ylabel(labels[i])
            ax.grid(True)
            self.lines.append(line)
        self.axs[-1].set_xlabel('Time [s]')
        self.fig.tight_layout()

        self.start_time = time.time()

        # use a separate thread for interactive matplotlib loop
        threading.Thread(target=self._matplotlib_loop, daemon=True).start()

        self.get_logger().info('soes_graph started — plotting /arm/joint_targets')

    def on_joint(self, msg: JointTargets):
        t = time.time() - self.start_time
        self.time_data.append(t)

        if msg.use_velocity:
            self.use_velocity = True
            values = msg.velocity
        else:
            values = msg.position

        for i in range(4):
            self.data[i].append(values[i])

    def _matplotlib_loop(self):
        plt.ion()
        while rclpy.ok():
            if len(self.time_data) > 2:
                x = list(self.time_data)
                for i in range(4):
                    y = list(self.data[i])
                    self.lines[i].set_data(x, y)
                    self.axs[i].relim()
                    self.axs[i].autoscale_view()

                ylabel = 'Angular velocity [rad/s]' if self.use_velocity else 'Angular position [rad]'
                for ax in self.axs:
                    ax.set_ylabel(ylabel)

                self.axs[-1].set_xlabel('Time [s]')
                plt.pause(0.01)
            else:
                plt.pause(0.05)
        plt.ioff()
        plt.show()


def main():
    rclpy.init()
    node = GraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
