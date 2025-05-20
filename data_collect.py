import time
import sys
import numpy as np
import zmq
import json
import threading
import cv2
import os
from datetime import datetime
from scipy import interpolate

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

import config as config
from hdf5_converter import convert_to_hdf5

class CombinedStateRecorder:
    def __init__(self):
        # Go2 robot dog variables
        self.low_state = None
        self.recording = False
        self.start_time = None
        
        # Auto-convert to HDF5 after recording
        self.auto_convert_hdf5 = config.AUTO_CONVERT_HDF5
        self.last_recording_dir = None
        
        # Go2 data storage - raw
        self.go2_joint_positions_raw = []
        self.go2_joint_velocities_raw = []
        self.go2_joint_torques_raw = []
        self.go2_imu_quaternions_raw = []
        self.go2_imu_gyroscopes_raw = []
        self.go2_imu_accelerometers_raw = []
        self.go2_timestamps_raw = []
        
        # D1 robotic arm data storage - raw
        self.d1_command_positions_raw = []    
        self.d1_command_timestamps_raw = []   
        self.d1_actual_positions_raw = []     
        self.d1_actual_timestamps_raw = []    
        
        # Camera frame storage - raw
        self.camera_frames_raw = []           
        self.camera_timestamps_raw = []       
        
        # Synchronized data storage
        self.sync_timestamps = []
        self.go2_joint_positions = []
        self.go2_joint_velocities = []
        self.go2_joint_torques = []
        self.go2_imu_quaternions = []
        self.go2_imu_gyroscopes = []
        self.go2_imu_accelerometers = []
        self.d1_command_positions = []
        self.d1_actual_positions = []
        self.camera_frames = []
        
        # Recording counters and status
        self.go2_record_count = 0
        self.d1_command_count = 0
        self.d1_actual_count = 0
        self.camera_frame_count = 0
        
        # ZMQ context and sockets
        self.zmq_context = zmq.Context()
        self.zmq_command_socket = None
        self.zmq_actual_socket = None
        self.zmq_camera_socket = None
        
        # Threads
        self.zmq_command_thread = None
        self.zmq_actual_thread = None
        self.zmq_camera_thread = None
        self.stop_event = threading.Event()
        
        # Synchronization
        self.sync_sample_rate = config.SYNC_RATE
        self.sync_interval = 1.0 / self.sync_sample_rate
        
        # Camera params
        self.save_camera_frames = True
        self.display_camera_frames = config.CAMERA_DISPLAY

    def Init(self):
        # Initialize Go2 subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler)
        
        # Initialize ZMQ connection for D1 arm commands (5555)
        self.zmq_command_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_command_socket.connect(config.D1_CMD_ADDRESS)
        self.zmq_command_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Initialize ZMQ connection for D1 arm actual positions (5556)
        self.zmq_actual_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_actual_socket.connect(config.D1_ACT_ADDRESS)
        self.zmq_actual_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Initialize ZMQ connection for duo cameras (5557)
        self.zmq_camera_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_camera_socket.connect(config.CAMERA_ADDRESS)
        self.zmq_camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_camera_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        
        print(f"All data will be synchronized to {self.sync_sample_rate}Hz")
        
        try:
            self.zmq_camera_socket.recv(flags=zmq.NOBLOCK)
            print("Camera connected successfully!")
        except zmq.Again:
            print("Waiting for camera data...")

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        
        # Record Go2 data
        if self.recording:
            self.RecordGo2Data(msg)
            # self.UpdateStatus()

    def RecordGo2Data(self, state):
        # Extract joint data
        positions = []
        velocities = []
        torques = []
        for i in range(12):
            positions.append(state.motor_state[i].q)
            velocities.append(state.motor_state[i].dq)
            torques.append(state.motor_state[i].tau_est)
        
        # Extract IMU data
        quaternion = state.imu_state.quaternion
        gyroscope = state.imu_state.gyroscope
        accelerometer = state.imu_state.accelerometer
        
        # Record all data
        self.go2_joint_positions_raw.append(positions)
        self.go2_joint_velocities_raw.append(velocities)
        self.go2_joint_torques_raw.append(torques)
        self.go2_imu_quaternions_raw.append(quaternion)
        self.go2_imu_gyroscopes_raw.append(gyroscope)
        self.go2_imu_accelerometers_raw.append(accelerometer)
        
        # Record timestamp
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.go2_timestamps_raw.append(elapsed_time)
        
        # Increment recording counter
        self.go2_record_count += 1

    def D1CommandThread(self):
        """Thread to receive and record D1 command positions."""
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    msg = self.zmq_command_socket.recv_string(flags=zmq.NOBLOCK)
                    data = json.loads(msg)
                    
                    if "positions" in data:
                        # Record joint positions and timestamp
                        self.d1_command_positions_raw.append(data["positions"])
                        self.d1_command_timestamps_raw.append(time.time() - self.start_time)
                        self.d1_command_count += 1
                except zmq.Again:
                    pass
                except json.JSONDecodeError:
                    pass

                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"D1 command thread error: {str(e)}")
        
        # End of thread
        print("D1 command thread ended")

    def D1ActualThread(self):
        """Thread to receive and record D1 actual positions."""
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    msg_raw = self.zmq_actual_socket.recv_string(flags=zmq.NOBLOCK)
                    if msg_raw.startswith("{\"joint_positions\":"):
                        try:
                            data = json.loads(msg_raw)
                            if "joint_positions" in data:
                                self.d1_actual_positions_raw.append(data["joint_positions"])
                                self.d1_actual_timestamps_raw.append(time.time() - self.start_time)
                                self.d1_actual_count += 1
                        except json.JSONDecodeError:
                            pass
                except zmq.Again:
                    pass
                
                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"D1 actual thread error: {str(e)}")
        
        # End of thread
        print("D1 actual thread ended")

    def CameraThread(self):
        """Thread to receive and record duo camera frames."""
        # For FPS calculation
        fps_frame_count = 0
        fps_start_time = time.time()
        fps = 0
        display_window_created = False
        
        # Main thread loop
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    jpg_buffer = self.zmq_camera_socket.recv(flags=zmq.NOBLOCK)
                    
                    img_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        current_time = time.time()
                        fps_frame_count += 1
                        fps_elapsed = current_time - fps_start_time
                        
                        if fps_elapsed >= 1.0:
                            fps = fps_frame_count / fps_elapsed
                            fps_frame_count = 0
                            fps_start_time = current_time
                        
                        # If saving frames is enabled
                        if self.save_camera_frames:
                            self.camera_frames_raw.append(frame.copy())
                            self.camera_timestamps_raw.append(current_time - self.start_time)
                            self.camera_frame_count += 1
                        
                        # If displaying frames is enabled
                        if self.display_camera_frames:
                            display_frame = frame.copy()
                            cv2.putText(
                                display_frame, 
                                f"Duo Cameras - Frame: {self.camera_frame_count} FPS: {fps:.1f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                            )
                            
                            if not display_window_created:
                                cv2.namedWindow('Duo Cameras', cv2.WINDOW_NORMAL)
                                display_window_created = True
                                
                            cv2.imshow('Duo Cameras', display_frame)
                            cv2.waitKey(1)
                    
                except zmq.Again:
                    pass
                
                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"Camera thread error: {str(e)}")
        
        print("Camera thread ended")

    def UpdateStatus(self):
        """Update recording status"""
        if self.recording:
            elapsed = time.time() - self.start_time
            go2_rate = self.go2_record_count / elapsed if elapsed > 0 else 0
            d1_cmd_rate = self.d1_command_count / elapsed if elapsed > 0 else 0
            d1_act_rate = self.d1_actual_count / elapsed if elapsed > 0 else 0
            cam_rate = self.camera_frame_count / elapsed if elapsed > 0 else 0
            
            status = f"\rRec: {elapsed:.1f}s | Go2: {self.go2_record_count} ({go2_rate:.1f}Hz) | "
            status += f"D1 cmd: {self.d1_command_count} ({d1_cmd_rate:.1f}Hz) | "
            status += f"D1 act: {self.d1_actual_count} ({d1_act_rate:.1f}Hz) | "
            status += f"Cameras: {self.camera_frame_count} ({cam_rate:.1f}fps)"
            
            print(status, end='', flush=True)

    def StartRecording(self):
        if self.recording:
            print("Already recording...")
            return
        
        # Clear previous data - raw
        self.go2_joint_positions_raw = []
        self.go2_joint_velocities_raw = []
        self.go2_joint_torques_raw = []
        self.go2_imu_quaternions_raw = []
        self.go2_imu_gyroscopes_raw = []
        self.go2_imu_accelerometers_raw = []
        self.go2_timestamps_raw = []
        
        self.d1_command_positions_raw = []
        self.d1_command_timestamps_raw = []
        self.d1_actual_positions_raw = []
        self.d1_actual_timestamps_raw = []
        
        self.camera_frames_raw = []
        self.camera_timestamps_raw = []
        
        # Clear previous data
        self.sync_timestamps = []
        self.go2_joint_positions = []
        self.go2_joint_velocities = []
        self.go2_joint_torques = []
        self.go2_imu_quaternions = []
        self.go2_imu_gyroscopes = []
        self.go2_imu_accelerometers = []
        self.d1_command_positions = []
        self.d1_actual_positions = []
        self.camera_frames = []
        
        # Reset counters
        self.go2_record_count = 0
        self.d1_command_count = 0
        self.d1_actual_count = 0
        self.camera_frame_count = 0
        
        # Start recording
        self.start_time = time.time()
        self.recording = True
        
        # Reset stop event and start threads
        self.stop_event.clear()
        
        # Start thread for D1 command positions
        self.zmq_command_thread = threading.Thread(target=self.D1CommandThread)
        self.zmq_command_thread.daemon = True
        self.zmq_command_thread.start()
        
        # Start thread for D1 actual positions
        self.zmq_actual_thread = threading.Thread(target=self.D1ActualThread)
        self.zmq_actual_thread.daemon = True
        self.zmq_actual_thread.start()
        
        # Start thread for combined cameras
        self.zmq_camera_thread = threading.Thread(target=self.CameraThread)
        self.zmq_camera_thread.daemon = True
        self.zmq_camera_thread.start()
        
        print("Starting data recording...")

    def StopRecording(self):
        if not self.recording:
            print("No active recording...")
            return
        
        # Stop recording
        self.recording = False
        self.stop_event.set()
        
        # Wait for threads to finish
        threads = [
            (self.zmq_command_thread, "D1 command"),
            (self.zmq_actual_thread, "D1 actual"),
            (self.zmq_camera_thread, "Camera")
        ]
        
        for thread, name in threads:
            if thread and thread.is_alive():
                print(f"Waiting for {name} thread to finish...")
                thread.join(timeout=2.0)
        
        # Calculate total elapsed time from raw data
        total_elapsed = 0
        if len(self.go2_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.go2_timestamps_raw[-1])
        if len(self.d1_command_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.d1_command_timestamps_raw[-1])
        if len(self.d1_actual_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.d1_actual_timestamps_raw[-1])
        if len(self.camera_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.camera_timestamps_raw[-1])
        
        # Calculate actual sampling rates from raw data
        go2_sample_rate = self.go2_record_count / total_elapsed if total_elapsed > 0 else 0
        d1_cmd_sample_rate = self.d1_command_count / total_elapsed if total_elapsed > 0 else 0
        d1_act_sample_rate = self.d1_actual_count / total_elapsed if total_elapsed > 0 else 0
        camera_frame_rate = self.camera_frame_count / total_elapsed if total_elapsed > 0 else 0
        
        print("")
        print(f"Recording stopped. Total time: {total_elapsed:.2f} seconds")
        print(f"Raw data collection rates:")
        print(f"- Go2: {self.go2_record_count} samples ({go2_sample_rate:.1f}Hz)")
        print(f"- D1 cmd: {self.d1_command_count} samples ({d1_cmd_sample_rate:.1f}Hz)")
        print(f"- D1 act: {self.d1_actual_count} samples ({d1_act_sample_rate:.1f}Hz)")
        print(f"- Cameras: {self.camera_frame_count} frames ({camera_frame_rate:.1f}fps)")
        
        # Synchronize data to common timeline
        print(f"Synchronizing all data to {self.sync_sample_rate}Hz timeline...")
        self.SynchronizeData()
        
        # Save data if any was recorded
        has_data = (self.go2_record_count > 0 or self.d1_command_count > 0 or 
                   self.d1_actual_count > 0 or self.camera_frame_count > 0)
        
        if has_data:
            recording_dir = self.SaveToNumpy()
            self.last_recording_dir = recording_dir
            
            # Auto convert to HDF5 if enabled
            if self.auto_convert_hdf5 and recording_dir:
                self.ConvertToHDF5(recording_dir)
        else:
            print("No data recorded, not saving file.")
            self.last_recording_dir = None

    def ConvertToHDF5(self, recording_dir):
        """Convert the recording directory to HDF5 format"""
        try:
            print(f"Converting {recording_dir} to HDF5 format...")
            success, hdf5_file = convert_to_hdf5(recording_dir)
            if success:
                print(f"Successfully converted to HDF5: {hdf5_file}")
                return hdf5_file
            else:
                print("HDF5 conversion failed.")
                return None
        except Exception as e:
            print(f"Error during HDF5 conversion: {str(e)}")
            return None

    def SynchronizeData(self):
        """Synchronize all data to a common timeline using interpolation"""
        # Check if we have any data to synchronize
        if not (len(self.go2_timestamps_raw) > 0 or len(self.d1_command_timestamps_raw) > 0 or 
                len(self.d1_actual_timestamps_raw) > 0 or len(self.camera_timestamps_raw) > 0):
            print("No data to synchronize")
            return
            
        # Find the earliest and latest timestamps across all data sources
        start_times = []
        end_times = []
        
        if len(self.go2_timestamps_raw) > 0:
            start_times.append(self.go2_timestamps_raw[0])
            end_times.append(self.go2_timestamps_raw[-1])
            
        if len(self.d1_command_timestamps_raw) > 0:
            start_times.append(self.d1_command_timestamps_raw[0])
            end_times.append(self.d1_command_timestamps_raw[-1])
            
        if len(self.d1_actual_timestamps_raw) > 0:
            start_times.append(self.d1_actual_timestamps_raw[0])
            end_times.append(self.d1_actual_timestamps_raw[-1])
            
        if len(self.camera_timestamps_raw) > 0:
            start_times.append(self.camera_timestamps_raw[0])
            end_times.append(self.camera_timestamps_raw[-1])
        
        # Use the latest start time and earliest end time to ensure all data is available
        sync_start = max(start_times) if start_times else 0
        sync_end = min(end_times) if end_times else 0
        
        if sync_end <= sync_start:
            print("WARNING: Invalid time range for synchronization")
            return
        
        # Create common timeline at common Hz
        self.sync_timestamps = np.arange(sync_start, sync_end, self.sync_interval)
        
        # Synchronize Go2 data if available
        if len(self.go2_timestamps_raw) > 1:
            print("Synchronizing Go2 data...")
            go2_timestamps = np.array(self.go2_timestamps_raw)
            
            # Interpolate joint positions
            if len(self.go2_joint_positions_raw) > 0:
                positions_array = np.array(self.go2_joint_positions_raw)
                self.go2_joint_positions = self._interpolate_array(go2_timestamps, positions_array, self.sync_timestamps)
                
            # Interpolate joint velocities
            if len(self.go2_joint_velocities_raw) > 0:
                velocities_array = np.array(self.go2_joint_velocities_raw)
                self.go2_joint_velocities = self._interpolate_array(go2_timestamps, velocities_array, self.sync_timestamps)
                
            # Interpolate joint torques
            if len(self.go2_joint_torques_raw) > 0:
                torques_array = np.array(self.go2_joint_torques_raw)
                self.go2_joint_torques = self._interpolate_array(go2_timestamps, torques_array, self.sync_timestamps)
                
            # Interpolate IMU quaternions
            if len(self.go2_imu_quaternions_raw) > 0:
                quaternions_array = np.array(self.go2_imu_quaternions_raw)
                self.go2_imu_quaternions = self._interpolate_array(go2_timestamps, quaternions_array, self.sync_timestamps)
                
            # Interpolate IMU gyroscopes
            if len(self.go2_imu_gyroscopes_raw) > 0:
                gyroscopes_array = np.array(self.go2_imu_gyroscopes_raw)
                self.go2_imu_gyroscopes = self._interpolate_array(go2_timestamps, gyroscopes_array, self.sync_timestamps)
                
            # Interpolate IMU accelerometers
            if len(self.go2_imu_accelerometers_raw) > 0:
                accelerometers_array = np.array(self.go2_imu_accelerometers_raw)
                self.go2_imu_accelerometers = self._interpolate_array(go2_timestamps, accelerometers_array, self.sync_timestamps)
        
        # Synchronize D1 command data if available
        if len(self.d1_command_timestamps_raw) > 1 and len(self.d1_command_positions_raw) > 0:
            print("Synchronizing D1 command data...")
            d1_cmd_timestamps = np.array(self.d1_command_timestamps_raw)
            d1_cmd_positions = np.array(self.d1_command_positions_raw)
            self.d1_command_positions = self._interpolate_array(d1_cmd_timestamps, d1_cmd_positions, self.sync_timestamps)
        
        # Synchronize D1 actual data if available
        if len(self.d1_actual_timestamps_raw) > 1 and len(self.d1_actual_positions_raw) > 0:
            print("Synchronizing D1 actual data...")
            d1_act_timestamps = np.array(self.d1_actual_timestamps_raw)
            d1_act_positions = np.array(self.d1_actual_positions_raw)
            self.d1_actual_positions = self._interpolate_array(d1_act_timestamps, d1_act_positions, self.sync_timestamps)
        
        # Synchronize camera frames if available
        if len(self.camera_timestamps_raw) > 1 and len(self.camera_frames_raw) > 0:
            print("Synchronizing camera frames...")
            self.camera_frames = self._nearest_frames(self.camera_timestamps_raw, self.camera_frames_raw, self.sync_timestamps)
        
        print(f"Synchronized {len(self.sync_timestamps)} samples at {self.sync_sample_rate}Hz")

    def _interpolate_array(self, src_timestamps, src_values, target_timestamps):
        """Interpolate array data to target timestamps"""
        if len(src_timestamps) != len(src_values):
            print(f"ERROR: Timestamp and value arrays have different lengths: {len(src_timestamps)} vs {len(src_values)}")
            return []
            
        result = []
        # If src_values is 1D, use simple interpolation
        if len(src_values.shape) == 1:
            interp_func = interpolate.interp1d(src_timestamps, src_values, axis=0, 
                                              bounds_error=False, fill_value="extrapolate")
            result = interp_func(target_timestamps)
        # If src_values is 2D, interpolate each component
        elif len(src_values.shape) == 2:
            try:
                interp_func = interpolate.interp1d(src_timestamps, src_values, axis=0, 
                                                bounds_error=False, fill_value="extrapolate")
                result = interp_func(target_timestamps)
            except Exception as e:
                print(f"Interpolation error: {e}")
                # Fallback method: use nearest neighbor for each point
                result = np.zeros((len(target_timestamps), src_values.shape[1]))
                for i, t in enumerate(target_timestamps):
                    idx = np.abs(src_timestamps - t).argmin()
                    result[i] = src_values[idx]
        else:
            print(f"ERROR: Unsupported array shape for interpolation: {src_values.shape}")
            
        return result

    def _nearest_frames(self, src_timestamps, src_frames, target_timestamps):
        """Select nearest camera frames for target timestamps"""
        result = []
        
        # Check if we have enough source frames
        if len(src_timestamps) < 1 or len(src_frames) < 1:
            return result
            
        # Convert source timestamps to numpy array if not already
        src_timestamps_array = np.array(src_timestamps)
        
        for target_time in target_timestamps:
            # Find index of nearest timestamp
            idx = np.abs(src_timestamps_array - target_time).argmin()
            # Add the corresponding frame to the result
            result.append(src_frames[idx].copy())
            
        return result

    def SaveToNumpy(self):
        # Generate timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate total recording duration from raw data
        total_duration = 0
        if len(self.go2_timestamps_raw) > 0:
            total_duration = max(total_duration, self.go2_timestamps_raw[-1])
        if len(self.d1_command_timestamps_raw) > 0:
            total_duration = max(total_duration, self.d1_command_timestamps_raw[-1])
        if len(self.d1_actual_timestamps_raw) > 0:
            total_duration = max(total_duration, self.d1_actual_timestamps_raw[-1])
        if len(self.camera_timestamps_raw) > 0:
            total_duration = max(total_duration, self.camera_timestamps_raw[-1])
        
        # Create directory for this recording session
        recording_dir = f"recording_{timestamp}"
        os.makedirs(recording_dir, exist_ok=True)
        
        # Create separate directories for raw and synchronized data
        raw_dir = os.path.join(recording_dir, "raw_data")
        sync_dir = os.path.join(recording_dir, "sync_data")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(sync_dir, exist_ok=True)
        
        # Save raw Go2 data if available
        if self.go2_record_count > 0:
            go2_raw_filename = os.path.join(raw_dir, "go2_raw.npz")
            
            # Convert to numpy arrays
            np_positions = np.array(self.go2_joint_positions_raw)
            np_velocities = np.array(self.go2_joint_velocities_raw)
            np_torques = np.array(self.go2_joint_torques_raw)
            np_quaternions = np.array(self.go2_imu_quaternions_raw)
            np_gyroscopes = np.array(self.go2_imu_gyroscopes_raw)
            np_accelerometers = np.array(self.go2_imu_accelerometers_raw)
            np_timestamps = np.array(self.go2_timestamps_raw)
            
            # Calculate actual sampling rate
            if len(self.go2_timestamps_raw) > 1:
                avg_sample_rate = (len(self.go2_timestamps_raw) - 1) / (self.go2_timestamps_raw[-1] - self.go2_timestamps_raw[0])
            else:
                avg_sample_rate = 0
            
            # Save as npz file
            np.savez(go2_raw_filename,
                    joint_positions=np_positions,
                    joint_velocities=np_velocities,
                    joint_torques=np_torques,
                    imu_quaternions=np_quaternions,
                    imu_gyroscopes=np_gyroscopes,
                    imu_accelerometers=np_accelerometers,
                    timestamps=np_timestamps,
                    metadata={
                        "recorded_at": datetime.now().isoformat(),
                        "duration": self.go2_timestamps_raw[-1] if len(self.go2_timestamps_raw) > 0 else 0,
                        "samples": len(self.go2_timestamps_raw),
                        "avg_sample_rate": avg_sample_rate
                    })
            
            print(f"Raw Go2 data saved to {go2_raw_filename}")
        
        # Save raw D1 data if available
        if self.d1_command_count > 0 or self.d1_actual_count > 0:
            d1_raw_filename = os.path.join(raw_dir, "d1_raw.npz")
            
            # Convert to numpy arrays
            d1_cmd_positions_array = np.array(self.d1_command_positions_raw) if self.d1_command_positions_raw else np.array([])
            d1_cmd_timestamps_array = np.array(self.d1_command_timestamps_raw) if self.d1_command_timestamps_raw else np.array([])
            d1_act_positions_array = np.array(self.d1_actual_positions_raw) if self.d1_actual_positions_raw else np.array([])
            d1_act_timestamps_array = np.array(self.d1_actual_timestamps_raw) if self.d1_actual_timestamps_raw else np.array([])
            
            # Calculate actual sampling rates
            if len(self.d1_command_timestamps_raw) > 1:
                d1_cmd_sample_rate = (len(self.d1_command_timestamps_raw) - 1) / (self.d1_command_timestamps_raw[-1] - self.d1_command_timestamps_raw[0])
            else:
                d1_cmd_sample_rate = 0
                
            if len(self.d1_actual_timestamps_raw) > 1:
                d1_act_sample_rate = (len(self.d1_actual_timestamps_raw) - 1) / (self.d1_actual_timestamps_raw[-1] - self.d1_actual_timestamps_raw[0])
            else:
                d1_act_sample_rate = 0
            
            # Save as npz file
            np.savez(d1_raw_filename,
                    command_positions=d1_cmd_positions_array,
                    command_timestamps=d1_cmd_timestamps_array,
                    actual_positions=d1_act_positions_array,
                    actual_timestamps=d1_act_timestamps_array,
                    metadata={
                        "recorded_at": datetime.now().isoformat(),
                        "command_samples": len(self.d1_command_timestamps_raw),
                        "actual_samples": len(self.d1_actual_timestamps_raw),
                        "command_sample_rate": d1_cmd_sample_rate,
                        "actual_sample_rate": d1_act_sample_rate
                    })
            
            print(f"Raw D1 data saved to {d1_raw_filename}")
        
        # Save raw camera frames if available
        if self.camera_frame_count > 0:
            # Create directory for raw camera frames
            raw_camera_dir = os.path.join(raw_dir, "camera_frames")
            os.makedirs(raw_camera_dir, exist_ok=True)
            
            # Save frames as JPEG files
            for i, frame in enumerate(self.camera_frames_raw):
                frame_file = os.path.join(raw_camera_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_file, frame)
            
            # Save timestamps
            camera_raw_timestamps_file = os.path.join(raw_dir, "camera_raw_timestamps.npy")
            np.save(camera_raw_timestamps_file, np.array(self.camera_timestamps_raw))
            
            # Calculate actual frame rate
            if len(self.camera_timestamps_raw) > 1:
                camera_fps = (len(self.camera_timestamps_raw) - 1) / (self.camera_timestamps_raw[-1] - self.camera_timestamps_raw[0])
            else:
                camera_fps = 0
                
            # Save metadata
            camera_raw_meta_file = os.path.join(raw_dir, "camera_raw_metadata.npz")
            np.savez(camera_raw_meta_file, 
                    frame_count=self.camera_frame_count,
                    avg_fps=camera_fps,
                    duration=self.camera_timestamps_raw[-1] if len(self.camera_timestamps_raw) > 0 else 0,
                    recorded_at=datetime.now().isoformat())
            
            print(f"Raw camera data saved to {raw_camera_dir} ({self.camera_frame_count} frames)")
        
        # Save synchronized data if available
        if len(self.sync_timestamps) > 0:
            # Save synchronized Go2 data
            if len(self.go2_joint_positions) > 0:
                go2_sync_filename = os.path.join(sync_dir, "go2_sync.npz")
                np.savez(go2_sync_filename,
                        joint_positions=self.go2_joint_positions,
                        joint_velocities=self.go2_joint_velocities,
                        joint_torques=self.go2_joint_torques,
                        imu_quaternions=self.go2_imu_quaternions,
                        imu_gyroscopes=self.go2_imu_gyroscopes,
                        imu_accelerometers=self.go2_imu_accelerometers,
                        timestamps=self.sync_timestamps,
                        metadata={
                            "recorded_at": datetime.now().isoformat(),
                            "duration": self.sync_timestamps[-1] - self.sync_timestamps[0],
                            "samples": len(self.sync_timestamps),
                            "sample_rate": self.sync_sample_rate
                        })
                print(f"Synchronized Go2 data saved to {go2_sync_filename}")
            
            # Save synchronized D1 data
            if len(self.d1_command_positions) > 0 or len(self.d1_actual_positions) > 0:
                d1_sync_filename = os.path.join(sync_dir, "d1_sync.npz")
                np.savez(d1_sync_filename,
                        command_positions=self.d1_command_positions if len(self.d1_command_positions) > 0 else [],
                        actual_positions=self.d1_actual_positions if len(self.d1_actual_positions) > 0 else [],
                        timestamps=self.sync_timestamps,
                        metadata={
                            "recorded_at": datetime.now().isoformat(),
                            "duration": self.sync_timestamps[-1] - self.sync_timestamps[0],
                            "samples": len(self.sync_timestamps),
                            "sample_rate": self.sync_sample_rate
                        })
                print(f"Synchronized D1 data saved to {d1_sync_filename}")
            
            # Save synchronized camera frames
            if len(self.camera_frames) > 0:
                # Create directory for synchronized camera frames
                sync_camera_dir = os.path.join(sync_dir, "camera_frames_sync")
                os.makedirs(sync_camera_dir, exist_ok=True)
                
                # Save frames as JPEG files
                for i, frame in enumerate(self.camera_frames):
                    frame_file = os.path.join(sync_camera_dir, f"frame_{i:06d}.jpg")
                    cv2.imwrite(frame_file, frame)
                
                # Also split each frame into top and bottom halves (camera 1 and camera 2)
                cam1_dir = os.path.join(sync_dir, "camera1_sync")
                cam2_dir = os.path.join(sync_dir, "camera2_sync")
                os.makedirs(cam1_dir, exist_ok=True)
                os.makedirs(cam2_dir, exist_ok=True)
                
                for i, frame in enumerate(self.camera_frames):
                    # Determine the midpoint
                    height = frame.shape[0]
                    mid = height // 2
                    
                    # Split the frame
                    cam1_frame = frame[:mid, :]
                    cam2_frame = frame[mid:, :]
                    
                    # Save individual camera frames
                    cam1_file = os.path.join(cam1_dir, f"frame_{i:06d}.jpg")
                    cam2_file = os.path.join(cam2_dir, f"frame_{i:06d}.jpg")
                    
                    cv2.imwrite(cam1_file, cam1_frame)
                    cv2.imwrite(cam2_file, cam2_frame)
                
                # Save timestamps
                camera_sync_timestamps_file = os.path.join(sync_dir, "camera_sync_timestamps.npy")
                np.save(camera_sync_timestamps_file, self.sync_timestamps)
                
                # Save metadata
                camera_sync_meta_file = os.path.join(sync_dir, "camera_sync_metadata.npz")
                np.savez(camera_sync_meta_file, 
                        frame_count=len(self.camera_frames),
                        sample_rate=self.sync_sample_rate,
                        duration=self.sync_timestamps[-1] - self.sync_timestamps[0],
                        recorded_at=datetime.now().isoformat())
                
                print(f"Synchronized camera data saved to {sync_camera_dir} ({len(self.camera_frames)} frames)")
        
        # Save global metadata file
        metadata_filename = os.path.join(recording_dir, "recording_metadata.npz")
        np.savez(metadata_filename,
                timestamp=timestamp,
                total_duration=total_duration,
                raw_go2_samples=self.go2_record_count,
                raw_d1_command_samples=self.d1_command_count,
                raw_d1_actual_samples=self.d1_actual_count,
                raw_camera_frames=self.camera_frame_count,
                sync_samples=len(self.sync_timestamps),
                sync_sample_rate=self.sync_sample_rate,
                recorded_at=datetime.now().isoformat())
        
        print(f"All recording data saved to directory: {recording_dir}")
        return recording_dir
    
    def Cleanup(self):
        if self.display_camera_frames:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(0.1)
        
        # Close ZMQ sockets
        sockets = [
            (self.zmq_command_socket, "D1 command"),
            (self.zmq_actual_socket, "D1 actual"),
            (self.zmq_camera_socket, "Camera")
        ]
        
        for socket, name in sockets:
            if socket:
                socket.close()
                print(f"Closed {name} socket")
        
        # Terminate ZMQ context
        if self.zmq_context:
            self.zmq_context.term()
            print("ZMQ context terminated")


def main():
    print(f"Data will be synchronized to {config.SYNC_RATE}Hz")
    
    # Initialize Unitree Go2 channel factory
    ChannelFactoryInitialize(0, config.NETWORK_INTERFACE)

    # Create and initialize recorder
    recorder = CombinedStateRecorder()
    recorder.Init()

    print("Ready, press Enter to start a new recording...")

    try:
        while True:
            # Wait for user to press Enter
            input()
            
            if not recorder.recording:
                # Start recording
                recorder.StartRecording()
                print("Recording... Press Enter to stop and save")
            else:
                # Stop recording
                recorder.StopRecording()
                print("\nReady, press Enter to start a new recording...")
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
        if recorder.recording:
            recorder.StopRecording()
        recorder.Cleanup()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Program terminated by user")
        
    sys.exit(0)


if __name__ == "__main__":
    main()
    