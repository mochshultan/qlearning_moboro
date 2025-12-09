#!/usr/bin/env python3

import signal
import sys
import threading
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class GraphSubscriber(Node):
    def __init__(self, window):
        super().__init__('qlearning_graph')
        self.window = window
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'result',
            self.data_callback,
            10
        )

    def data_callback(self, msg):
        self.window.receive_data(msg)


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('Q-Learning Results')
        self.setGeometry(50, 50, 600, 650)

        self.episodes = []
        self.scores = []
        self.avg_scores = []
        self.count = 0

        self.plot()

        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()

    def receive_data(self, msg):
        self.count += 1
        self.episodes.append(self.count)
        self.scores.append(msg.data[0])
        
        # Hitung cumulative average
        avg = np.mean(self.scores)
        self.avg_scores.append(avg)

    def plot(self):
        self.scorePlt = pyqtgraph.PlotWidget(self, title='Episode Score')
        self.scorePlt.setGeometry(0, 10, 600, 300)

        self.avgPlt = pyqtgraph.PlotWidget(self, title='Average Score')
        self.avgPlt.setGeometry(0, 320, 600, 300)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        self.scorePlt.showGrid(x=True, y=True)
        self.avgPlt.showGrid(x=True, y=True)

        self.scorePlt.plot(self.episodes, self.scores, pen=(255, 0, 0), clear=True)
        self.avgPlt.plot(self.episodes, self.avg_scores, pen=(0, 255, 0), clear=True)

    def closeEvent(self, event):
        if self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    win = Window()

    def shutdown_handler(sig, frame):
        print('shutdown')
        if win.ros_subscriber is not None:
            win.ros_subscriber.destroy_node()
        rclpy.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
