#!/usr/bin/env python3
import time
import numpy as np
import zmq
from STservo_sdk import *

# Default normalization and joint limits
DEFAULT_NORMALIZATION_LIMITS = np.array([
    [481, 3696],
    [1151, 2706],
    [2330, 812],
    [767, 3295],
    [1533, 3623],
    [919, 3528],
    [2125, 1881],
])
DEFAULT_JOINT_LIMITS = np.radians(np.array([
    [-235, 35],
    [0, 135],
    [-135, 0],
    [-202.5, 22.5],
    [-90, 90],
    [-202.5, 22.5],
    [180, -180],
]))

class GelloController:
    def __init__(self, device_name: str, baud_rate: int = 1000000):
        self.num_motors = 7
        self.motor_pos = np.zeros(self.num_motors, dtype=np.float32)
        self.motor_speed = np.zeros(self.num_motors, dtype=np.float32)
        self.portOpen = False
        try:
            self.portHandler = PortHandler(device_name)
            self.packetHandler = sts(self.portHandler)
            if self.portHandler.openPort() and self.portHandler.setBaudRate(baud_rate):
                self.portOpen = True
            else:
                print("Failed to open port or set baud rate, using simulated data")
            self.groupSyncRead = GroupSyncRead(self.packetHandler, STS_PRESENT_POSITION_L, 4)
        except Exception as e:
            print(f"Controller init error: {e}, using simulated data")

    def read_joints(self):
        if not self.portOpen:
            t = time.time()
            self.motor_pos = 2000 + 500 * np.sin(t * 0.5 + np.arange(self.num_motors) * 0.5)
            return self.motor_pos
        try:
            for i in range(1, self.num_motors + 1):
                self.groupSyncRead.addParam(i)
            self.groupSyncRead.txRxPacket()
            for i in range(1, self.num_motors + 1):
                available, _ = self.groupSyncRead.isAvailable(i, STS_PRESENT_POSITION_L, 4)
                if available:
                    pos = self.groupSyncRead.getData(i, STS_PRESENT_POSITION_L, 2)
                    self.motor_pos[i-1] = pos
            self.groupSyncRead.clearParam()
        except Exception as e:
            print(f"Read joints error: {e}, using simulated data")
            t = time.time()
            self.motor_pos = 2000 + 500 * np.sin(t * 0.5 + np.arange(self.num_motors) * 0.5)
        return self.motor_pos

class GelloControllerWrapper:
    def __init__(self, controller: GelloController):
        self.ctrl = controller
        self.norm_limits = DEFAULT_NORMALIZATION_LIMITS
        self.joint_limits = DEFAULT_JOINT_LIMITS

    def get_joint_pos(self):
        raw = self.ctrl.read_joints()
        normed = (raw - self.norm_limits[:,1]) / (self.norm_limits[:,0] - self.norm_limits[:,1])
        return self.joint_limits[:,0] + normed * (self.joint_limits[:,1] - self.joint_limits[:,0])
    
def map_position_helper(x, ctrl_max, ctrl_min, mech_max, mech_min):
    return (x - ctrl_min) / (ctrl_max - ctrl_min) * (mech_max - mech_min) + mech_min

def map_position(controller_pos):
    J0, J1, J2, J3, J4, J5, J6 = controller_pos

    J0 = map_position_helper(J0, 0.5, -4,  -135,  135)
    J1 = map_position_helper(J1, 2.5, 0.58, -90,   90)
    J2 = map_position_helper(J2, 0,   -2.3, -90,   90)
    J3 = map_position_helper(J3, 0.85,  -2.8, -135, 135)
    J4 = map_position_helper(J4, 1.5, -1.62, -90,   90)
    J5 = map_position_helper(J5, -2,  -3.5, -135, 135)
    J6 = map_position_helper(J6, -5,  -17,   0,    65)

    return [J0, J1, J2, J3, J4, 0, J6]

def clamp_delta_move(current, target, max_delta_deg):
    current = np.array(current)
    target = np.array(target)
    delta = target - current

    max_delta = np.clip(delta, -max_delta_deg, max_delta_deg)
    next_pos = current + max_delta
    return next_pos.tolist()

def main():
    serial_port = "/dev/ttyACM0"
    zmq_endpoint = "tcp://*:5555"
    send_interval = 0.05
    max_speed_deg_per_step = 5

    controller = GelloController(device_name=serial_port)
    wrapper = GelloControllerWrapper(controller)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(zmq_endpoint)
    time.sleep(1)

    print(f"Publishing joint angles on {zmq_endpoint} with limited velocity...")

    current_pos = None
    try:
        while True:
            ctrl_pos = wrapper.get_joint_pos()
            target_pos = map_position(ctrl_pos)

            if current_pos is None:
                current_pos = target_pos

            current_pos = clamp_delta_move(current_pos, target_pos, max_speed_deg_per_step)

            pub.send_json({"positions": current_pos})
            time.sleep(send_interval)
    except KeyboardInterrupt:
        print("Shutting down publisher")
    finally:
        pub.close()
        ctx.term()

if __name__ == '__main__':
    main()