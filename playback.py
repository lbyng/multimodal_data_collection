import time
import sys
import numpy as np
import zmq
import json
import threading
import h5py
import os
import select
import argparse
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

class HDF5Replayer:
    def __init__(self):
        # Go2 robot control parameters
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        
        # HDF5 data
        self.hdf5_data = None
        self.actions = None
        self.num_samples = 0
        self.current_index = 0
        
        # Replay control
        self.start_time = None
        self.is_playing = False
        self.stop_event = threading.Event()
        
        # Rate control
        self.data_dt = 1/60.0  # Base time interval
        self.arm_dt = 1/60.0   # Arm time interval
        self.next_arm_time = 0
        
        # Go2 thread
        self.go2_cmdThreadPtr = None
        self.crc = CRC()
        
        # D1 ZMQ
        self.zmq_context = None
        self.zmq_socket = None
        
        # Time control 
        self.replay_speed = 1.0  # Replay speed multiplier

    def init(self):
        print("Initializing HDF5 replayer...")
        
        # Initialize Go2 robot
        self.init_low_cmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        
        status, result = self.msc.CheckMode()
        while result['name']:
            print(f"Releasing current mode: {result['name']}")
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
            
        # Initialize D1 mechanical arm ZMQ
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind("tcp://*:5555")
        
        print("Initialization complete")
        print("ZMQ publisher bound to tcp://*:5555")

    def init_low_cmd(self):
        """Initialize low-level command structure"""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def load_hdf5(self, filename):
        """Load HDF5 file"""
        try:
            print(f"Loading HDF5 file: {filename}")
            self.hdf5_data = h5py.File(filename, 'r')
            
            if 'action' not in self.hdf5_data:
                print("Error: No action data in HDF5 file")
                return False
                
            self.actions = self.hdf5_data['action'][:]
            self.num_samples = self.actions.shape[0]
            
            action_width = self.actions.shape[1]
            has_d1_data = action_width > 60
            
            print(f"Successfully loaded HDF5 data: {self.num_samples} samples")
            print(f"Action data dimensions: {self.actions.shape}")
            print(f"Contains D1 data: {has_d1_data}")
            
            if has_d1_data:
                print(f"D1 degrees of freedom: {action_width - 60}")
            
            duration = self.num_samples * self.data_dt
            print(f"Data duration: {duration:.2f} seconds")
            
            self.current_index = 0
            return True
            
        except Exception as e:
            print(f"Error loading HDF5 data: {e}")
            return False

    def start_replay(self, speed=1.0):
        """Start replaying data"""
        if self.hdf5_data is None:
            print("Error: Please load HDF5 data first")
            return False
            
        print("Preparing to start replay...")
        print("Warning: Make sure there are no obstacles around the robot and it's ready to move!")
        input("Press Enter to start replay...")
            
        self.is_playing = True
        self.stop_event.clear()
        self.start_time = time.time()
        self.next_arm_time = 0
        self.current_index = 0
        self.replay_speed = speed
        
        # Start Go2 thread
        self.go2_cmdThreadPtr = RecurrentThread(
            interval=1.0/30.0,
            target=self.go2_command_thread,
            name="go2_replay_cmd_thread"
        )
        self.go2_cmdThreadPtr.Start()
        
        # Start D1 thread
        self.d1_thread = threading.Thread(target=self.d1_replay_thread, name="d1_replay_thread")
        self.d1_thread.daemon = True
        self.d1_thread.start()
        
        print(f"Replay started, speed: {speed}x")
        return True
            
    def stop_replay(self):
        """Stop replay"""
        if not self.is_playing:
            return
        
        print("Stopping replay...")
        
        self.is_playing = False
        self.stop_event.set()
        
        # Wait for threads to complete
        if self.d1_thread and self.d1_thread.is_alive():
            self.d1_thread.join(timeout=2.0)
        
        # Switch Go2 to AI mode
        self.msc.SelectMode('ai')
        print("AI motion mode enabled")
        time.sleep(5)
        # self.sc.StandDown()
        # print("StandDown Enabled")
        
        print("Replay stopped")

    def go2_command_thread(self):
        if not self.is_playing or self.current_index >= self.num_samples:
            return
               
        # Get action data for current index
        current_action = self.actions[self.current_index]
        
        # Extract Go2 control commands
        positions = current_action[:12]
        velocities = current_action[12:24]
        torques = current_action[24:36]
        kp_values = current_action[36:48]
        kd_values = current_action[48:60]
        
        # Update commands
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = positions[i]
            self.low_cmd.motor_cmd[i].dq = velocities[i]
            self.low_cmd.motor_cmd[i].kp = kp_values[i]
            self.low_cmd.motor_cmd[i].kd = kd_values[i]
            self.low_cmd.motor_cmd[i].tau = torques[i]
            
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)
        
        elapsed_time = time.time() - self.start_time
        target_index = int(elapsed_time * (1.0 / self.data_dt) * self.replay_speed)
        
        if target_index > self.current_index:
            self.current_index = min(target_index, self.num_samples - 1)
            
            if self.current_index % 100 == 0:
                progress = (self.current_index / self.num_samples) * 100
                print(f"\rReplay progress: {progress:.1f}% (index: {self.current_index}/{self.num_samples})", end="", flush=True)
        
        if self.current_index >= self.num_samples - 1:
            print("\nReplay complete")
            self.stop_replay()

    def d1_replay_thread(self):
        """D1 mechanical arm replay thread function (max 20Hz)"""
        print("D1 replay thread started")
        has_d1_data = self.actions.shape[1] > 60
        
        if not has_d1_data:
            print("No D1 mechanical arm data in HDF5 file, skipping D1 replay")
            return
        
        d1_dof = self.actions.shape[1] - 60
        
        try:
            last_sent_index = -1
            
            while self.is_playing and self.current_index < self.num_samples:
                elapsed_time = time.time() - self.start_time
                
                if elapsed_time >= self.next_arm_time:
                    current_index = self.current_index
                    
                    if current_index > last_sent_index:
                        d1_command = self.actions[current_index, 60:60+d1_dof].tolist()
                        
                        msg = {"positions": d1_command}
                        try:
                            self.zmq_socket.send_json(msg)
                            last_sent_index = current_index
                        except Exception as e:
                            print(f"Error sending D1 command: {e}")
                    
                    self.next_arm_time = self.next_arm_time + (self.arm_dt / self.replay_speed)
                
                time.sleep(0.001)
                
                if self.stop_event.is_set():
                    break
            
        except Exception as e:
            print(f"D1 replay thread error: {e}")
        finally:
            print("D1 replay thread ended")

    def cleanup(self):
        """Clean up resources"""
        # Close HDF5 file
        if self.hdf5_data is not None:
            self.hdf5_data.close()
        
        # Close ZMQ socket
        if self.zmq_socket:
            self.zmq_socket.close()
        
        # Terminate ZMQ context
        if self.zmq_context:
            self.zmq_context.term()


def main():
    parser = argparse.ArgumentParser(description='HDF5 Replayer - Replay robot actions from HDF5 file')
    parser.add_argument('hdf5_file', type=str, help='Path to HDF5 file to replay')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed multiplier (default: 1.0)')
    parser.add_argument('--interface', type=str, default='eno1', help='Network interface (default: eno1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_file):
        print(f"Error: HDF5 file not found {args.hdf5_file}")
        sys.exit(1)
    
    ChannelFactoryInitialize(0, args.interface)
    
    replayer = HDF5Replayer()
    replayer.init()
    
    if not replayer.load_hdf5(args.hdf5_file):
        print("Failed to load data, exiting")
        replayer.cleanup()
        sys.exit(1)
    
    try:
        replayer.start_replay(speed=args.speed)
        
        while replayer.is_playing:
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() == 'q':
                    print("User requested to stop replay")
                    replayer.stop_replay()
                    break
                elif line.startswith('speed '):
                    try:
                        new_speed = float(line.split(' ')[1])
                        if new_speed > 0:
                            print(f"Adjusting replay speed to {new_speed}x")
                            replayer.replay_speed = new_speed
                    except:
                        print("Incorrect speed setting format, use 'speed 1.5' to adjust speed")
        
        print("Replay complete")
        print("Press Enter to exit")
        input()
        
    except KeyboardInterrupt:
        print("\nUser interrupt")
        if replayer.is_playing:
            replayer.stop_replay()
    finally:
        replayer.cleanup()
        print("Program exit")

if __name__ == '__main__':
    main()