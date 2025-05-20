import time
import pyspacemouse
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

class spacemouse_go2:
    def __init__(self):
        self.success = None
        self.state = None

        self.sport_client = None
        self.msc = None

        self.spacemouse_pos = [0.0, 0.0, 0.0]
        self.mapped_spacemouse_pos = [0.0, 0.0, 0.0]
        self.buttons = [0, 0]
        self.if_Damp = 0

        return None

    def connect_spacemouse(self):
        self.success = pyspacemouse.open()
        self.state = pyspacemouse.read()
        print("Spacemouse connected")
        return None
    
    def connect_robot(self):
        ChannelFactoryInitialize(0, "eno1")
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        print("Go2 Robot connected")

        self.msc = MotionSwitcherClient()
        self.msc.SelectMode("Normal")
        print("Normal motion mode enabled")
        return None
    
    def update_spacemouse(self):
        self.state = pyspacemouse.read()
        self.spacemouse_pos[0] = self.state.x
        self.spacemouse_pos[1] = self.state.y
        self.spacemouse_pos[2] = self.state.yaw
        self.mapped_spacemouse_pos = map_position(self.spacemouse_pos)

        self.buttons[0] = self.state.buttons[0]
        self.buttons[1] = self.state.buttons[1]

        self.if_damp = self.state.z
        return None

    def update_robot(self):
        self.sport_client.Move(self.mapped_spacemouse_pos[1], 
                               self.mapped_spacemouse_pos[0],
                               self.mapped_spacemouse_pos[2])
        if self.buttons[1]:
            self.sport_client.BalanceStand()
        
        if self.buttons[0]:
            self.sport_client.StandDown()

        if self.if_damp == -1.0:
            self.sport_client.Damp()
        return None 


def map_position_helper(x, ctrl_max, ctrl_min, mech_max, mech_min):
    return (x - ctrl_min) / (ctrl_max - ctrl_min) * (mech_max - mech_min) + mech_min

def map_position(spacemouse_pos):
    x, y, yaw = spacemouse_pos

    deadzone = 0.1
    if abs(x) < deadzone: x = 0
    if abs(y) < deadzone: y = 0
    if abs(yaw) < deadzone: yaw = 0

    x = map_position_helper(x, -0.6, 0.6, 0.6, -0.6)
    y = map_position_helper(y, -0.6, 0.6, -0.4, 0.4)
    yaw = map_position_helper(yaw, -1.0, 1.0, 0.8,  -0.8)

    return [x, y, yaw]

def main():

    robot = spacemouse_go2()
    robot.connect_spacemouse()
    robot.connect_robot()

    while True:
        robot.update_spacemouse()
        robot.update_robot()
        time.sleep(0.002)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")