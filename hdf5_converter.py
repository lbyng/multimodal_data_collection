import h5py
import numpy as np
import os
from tqdm import tqdm
import cv2
import glob
import argparse
from datetime import datetime

def convert_to_hdf5(recording_dir, output_file=None):
    if output_file is None:
        dir_name = os.path.basename(recording_dir)
        if dir_name.startswith("recording_"):
            timestamp = dir_name[len("recording_"):]  # Extract timestamp part
            output_file = os.path.join(os.path.dirname(recording_dir), f"{timestamp}.hdf5")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(os.path.dirname(recording_dir), f"{timestamp}.hdf5")
    
    # Check if recording directory exists
    if not os.path.exists(recording_dir):
        print(f"Recording directory {recording_dir} does not exist.")
        return False, None
    
    # Check if synchronized data exists
    sync_dir = os.path.join(recording_dir, "sync_data")
    if not os.path.exists(sync_dir):
        print(f"Synchronized data directory {sync_dir} does not exist.")
        return False, None
    
    # Load Go2 synchronized data
    go2_sync_file = os.path.join(sync_dir, "go2_sync.npz")
    if not os.path.exists(go2_sync_file):
        print(f"Go2 synchronized data file {go2_sync_file} not found.")
        return False, None
    
    print(f"Loading Go2 data from {go2_sync_file}...")
    go2_data = np.load(go2_sync_file, allow_pickle=True)
    
    # Load D1 synchronized data
    d1_sync_file = os.path.join(sync_dir, "d1_sync.npz")
    d1_data = None
    if os.path.exists(d1_sync_file):
        print(f"Loading D1 data from {d1_sync_file}...")
        d1_data = np.load(d1_sync_file, allow_pickle=True)
    else:
        print("D1 synchronized data not found. Proceeding without D1 data.")
    
    # Check for camera data
    cam1_dir = os.path.join(sync_dir, "camera1_sync")
    cam2_dir = os.path.join(sync_dir, "camera2_sync")
    
    has_camera1 = os.path.exists(cam1_dir) and len(os.listdir(cam1_dir)) > 0
    has_camera2 = os.path.exists(cam2_dir) and len(os.listdir(cam2_dir)) > 0
    
    if has_camera1:
        print(f"Found camera1 data in {cam1_dir}")
    else:
        print("Camera1 data not found.")
    
    if has_camera2:
        print(f"Found camera2 data in {cam2_dir}")
    else:
        print("Camera2 data not found.")
    
    # Get common timestamps and dataset size
    timestamps = go2_data['timestamps']
    num_samples = len(timestamps)
    
    print(f"Total synchronized samples: {num_samples}")
    
    # Create output HDF5 file
    print(f"Creating HDF5 file: {output_file}")
    with h5py.File(output_file, 'w') as f:
        # Create groups for organization
        obs_group = f.create_group('observations')
        motor_group = obs_group.create_group('motor_states')
        imu_group = obs_group.create_group('imu')
        
        # Add metadata attributes
        f.attrs['sim'] = False
        f.attrs['compress'] = False
        
        # Joint positions (qpos)
        joint_positions = go2_data['joint_positions']
        obs_group.create_dataset('qpos', data=joint_positions)
        
        # Joint velocities (qvel)
        joint_velocities = go2_data['joint_velocities']
        obs_group.create_dataset('qvel', data=joint_velocities)
        
        # Joint positions in motor space
        motor_group.create_dataset('q', data=joint_positions)
        
        # Joint velocities in motor space
        motor_group.create_dataset('dq', data=joint_velocities)
        
        # Joint torques
        motor_group.create_dataset('tau_est', data=go2_data['joint_torques'])
        
        # Orientation (quaternion)
        imu_group.create_dataset('orientation', data=go2_data['imu_quaternions'])
        
        # Angular velocity
        imu_group.create_dataset('angular_velocity', data=go2_data['imu_gyroscopes'])
        
        # Linear acceleration
        imu_group.create_dataset('linear_acceleration', data=go2_data['imu_accelerometers'])
        
        default_kp = 70.0
        default_kd = 5.0
        
        if d1_data is not None and 'command_positions' in d1_data and len(d1_data['command_positions']) > 0:
            d1_command_positions = d1_data['command_positions']
            d1_dof = d1_command_positions.shape[1]
            combined_action_data = np.zeros((num_samples, 12*5 + d1_dof), dtype=np.float32)
            
            combined_action_data[:, :12] = joint_positions  # positions (q)
            combined_action_data[:, 12:24] = joint_velocities  # velocities (dq)
            combined_action_data[:, 24:36] = go2_data['joint_torques']  # torques (tau)
            
            combined_action_data[:, 36:48] = default_kp
            combined_action_data[:, 48:60] = default_kd
            
            combined_action_data[:, 60:60+d1_dof] = d1_command_positions
            
            f.create_dataset('action', data=combined_action_data)
            print(f"Added combined Go2 (12 motors with control params) and D1 arm data as actions with shape {combined_action_data.shape}")
            
        else:
            go2_action_data = np.zeros((num_samples, 12*5), dtype=np.float32)
            
            go2_action_data[:, :12] = joint_positions  # positions (q)
            go2_action_data[:, 12:24] = joint_velocities  # velocities (dq)
            go2_action_data[:, 24:36] = go2_data['joint_torques']  # torques (tau)
            
            go2_action_data[:, 36:48] = default_kp
            go2_action_data[:, 48:60] = default_kd
            
            f.create_dataset('action', data=go2_action_data)
            print(f"Added Go2 motor data (12 motors with control params) as actions with shape {go2_action_data.shape}")
        
        # Process camera images if available
        if has_camera1 or has_camera2:
            images_group = obs_group.create_group('images')
            
            # Function to get all image files sorted by frame number
            def get_sorted_images(directory):
                image_files = glob.glob(os.path.join(directory, "frame_*.jpg"))
                image_files.sort(key=lambda x: int(os.path.basename(x).replace("frame_", "").replace(".jpg", "")))
                return image_files
            
            # Function to process and add images to HDF5 file
            def add_camera_images(camera_dir, camera_name):
                image_files = get_sorted_images(camera_dir)
                
                if not image_files:
                    print(f"No image files found in {camera_dir}")
                    return
                
                # Read first image to get dimensions
                sample_img = cv2.imread(image_files[0])
                height, width, channels = sample_img.shape
                
                # Create dataset for this camera
                camera_dataset = images_group.create_dataset(
                    camera_name, 
                    shape=(num_samples, height, width, channels),
                    dtype=np.uint8
                )
                
                # Read and store all images
                print(f"Adding {len(image_files)} {camera_name} images...")
                for i, img_file in enumerate(tqdm(image_files)):
                    if i >= num_samples:
                        break
                    img = cv2.imread(img_file)
                    # Convert BGR to RGB (OpenCV loads as BGR, but we want RGB for training)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    camera_dataset[i] = img
            
            # Add camera1 images
            if has_camera1:
                add_camera_images(cam1_dir, 'camera1')
            
            # Add camera2 images
            if has_camera2:
                add_camera_images(cam2_dir, 'camera2')
    
    print(f"Successfully converted {recording_dir} to {output_file}")
    return True, output_file


def convert_multiple_recordings(base_dir, output_dir=None):
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(base_dir, "hdf5_data")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all recording directories
    recording_dirs = glob.glob(os.path.join(base_dir, "recording_*"))
    recording_dirs.sort()
    
    if not recording_dirs:
        print(f"No recording directories found in {base_dir}")
        return []
    
    print(f"Found {len(recording_dirs)} recording directories")
    
    # Convert each recording directory
    output_files = []
    for rec_dir in recording_dirs:
        # Extract timestamp from directory name
        dir_name = os.path.basename(rec_dir)
        timestamp = dir_name[len("recording_"):]  # Extract timestamp part
        output_file = os.path.join(output_dir, f"{timestamp}.hdf5")
        
        print(f"Converting {dir_name} to {timestamp}.hdf5...")
        success, file_path = convert_to_hdf5(rec_dir, output_file)
        if success:
            output_files.append(file_path)
    
    print(f"Converted {len(output_files)} recordings to HDF5 format in {output_dir}")
    return output_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert synchronized recordings to ACT-compatible HDF5 format")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input directory (single recording directory or base directory containing multiple recordings)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (for single recording) or directory (for multiple recordings)")
    parser.add_argument("--multi", action="store_true",
                        help="Process multiple recordings in the input directory")
    
    args = parser.parse_args()
    
    if args.multi:
        convert_multiple_recordings(args.input, args.output)
    else:
        # Let convert_to_hdf5 handle automatic timestamp naming
        convert_to_hdf5(args.input, args.output)