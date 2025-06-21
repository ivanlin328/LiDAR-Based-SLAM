import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

from part1 import compute_trajectory
from pr2_utils import bresenham2D, plot_map
from load_data import load_dataset
from scan_matching2 import (
    lidar_scan_to_pc,
    odom_relative_transform,   # Make sure this function returns a 3×3 matrix
    get_pose_at_timestamp,
    ICP,
    apply_transform_2d         # Only accepts a 3×3 matrix
)
from tqdm import tqdm

def compute_depth_from_disparity(d):
    d_d = -0.00304 * d + 3.31
    depth = 1.03 / d_d
    return depth

def rgb_index_from_disparity(i, j, d_d, height, width):
    rgbi = (526.37 * i + 19276 - 7877.07 * d_d) / 585.051
    rgbj = (526.37 * j + 16662) / 585.051
    rgbi = int(np.clip(rgbi, 0, height - 1))
    rgbj = int(np.clip(rgbj, 0, width - 1))
    return rgbi, rgbj

def compute_camera_point(i, j, depth, intrinsic):
    x = (j - intrinsic['c_u']) * depth / intrinsic['f_u']
    y = (i - intrinsic['c_v']) * depth / intrinsic['f_v']
    z = depth
    return np.array([x, y, z])

def get_world_point(xyz_cam, T_robot_cam, T_world_robot):
    """
    camera(3D) -> robot -> world

    T_robot_cam: 4x4 transform (camera -> robot)
    T_world_robot: 4x4 transform (robot -> world)
    """
    pt_h = np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0])
    pt_robot = T_robot_cam @ pt_h
    pt_world = T_world_robot @ pt_robot
    return pt_world[:3]

def build_floor_color_map(points_world, colors, MAP, z_threshold=0.3):
    color_map = np.zeros((MAP['size'][0], MAP['size'][1], 3), dtype=np.uint8)
    
    for idx in range(len(points_world)):
        xw, yw, zw = points_world[idx]
        r, g, b = colors[idx]
        
        # Skip points that are not within the z threshold of 2.2
        if abs(zw - 2.2) > z_threshold:
            continue
        
        cx = int(np.floor((xw - MAP['min'][0]) / MAP['res'][0]))
        cy = int(np.floor((yw - MAP['min'][1]) / MAP['res'][1]))
        
        if (0 <= cx < MAP['size'][0]) and (0 <= cy < MAP['size'][1]):
            color_map[cx, cy, :] = (r, g, b)
    
    return color_map

def get_transformation_matrix(translation, rotation_angles):
    roll, pitch, yaw = rotation_angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T

def build_2D_transform(x, y, theta):
    """
    Helper to build a 4x4 transform from (x, y, theta)
    for the robot's pose in the world.
    """
    T = np.eye(4)
    c = np.cos(theta)
    s = np.sin(theta)
    T[0,0] = c;  T[0,1] = -s
    T[1,0] = s;  T[1,1] =  c
    T[0,3] = x;  T[1,3] = y
    return T

def compute_icp_trajectory(encoder_counts, encoder_stamps, imu_stamps,
                           yaw_rate, lidar_ranges, lidar_stamps,
                           lidar_angle_min, lidar_angle_increment,
                           lidar_range_min, lidar_range_max):
    FR = encoder_counts[0, :]
    FL = encoder_counts[1, :]
    RR = encoder_counts[2, :]
    RL = encoder_counts[3, :]

    x, y, theta = compute_trajectory(FR, FL, RR, RL, yaw_rate, encoder_stamps, imu_stamps)
    theta = np.unwrap(theta)

    n_scans = lidar_ranges.shape[1]
    scan0 = lidar_ranges.T[0]
    pc0 = lidar_scan_to_pc(scan0, lidar_angle_min, lidar_angle_increment,
                           lidar_range_min, lidar_range_max)
    pcs = [pc0]
    T_global = [np.eye(3)]
    
    for i in tqdm(range(n_scans - 1), desc='Processing scans'):
        # Get the previous and current LiDAR scans
        scan_prev = lidar_ranges.T[i]
        scan_curr = lidar_ranges.T[i+1]
        points_prev = lidar_scan_to_pc(scan_prev, lidar_angle_min, lidar_angle_increment,
                                       lidar_range_min, lidar_range_max)
        points_curr = lidar_scan_to_pc(scan_curr, lidar_angle_min, lidar_angle_increment,
                                       lidar_range_min, lidar_range_max)
        
        # Find the corresponding odometry pose using LiDAR timestamps
        t_prev = lidar_stamps[i]
        t_curr = lidar_stamps[i+1]
        x_prev, y_prev, th_prev = get_pose_at_timestamp(t_prev, encoder_stamps, x, y, theta)
        x_curr, y_curr, th_curr = get_pose_at_timestamp(t_curr, encoder_stamps, x, y, theta)
        
        # (a) Use odometry to compute the 3x3 relative pose T_odom
        T_odom = odom_relative_transform(x_prev, y_prev, th_prev, x_curr, y_curr, th_curr)
        
        # (b) Accumulate from the previous global pose T_global[-1] to obtain the initial pose T_init
        T_init = T_global[-1] @ T_odom
        
        # (c) Transform the current point cloud to global coordinates (2D) using T_init
        pc_curr_init = apply_transform_2d(points_curr, T_init)
        
        # (d) Prepare the ICP target: previous point cloud transformed to global (2D)
        pc_prev_global = apply_transform_2d(points_prev, T_global[-1])
        
        # (e) Call ICP with an initial estimate of identity (3x3)
        T_icp = ICP(pc_curr_init, pc_prev_global,
                    init_odometry=np.eye(3),
                    max_iterations=20, tolerance=1e-7, distance_threshold=0.1)
        
        # (f) Obtain the new global pose
        T_new = T_icp @ T_init
        T_global.append(T_new)
        
        # (g) Transform the current point cloud to global coordinates for subsequent map updating
        pc_curr_global = apply_transform_2d(points_curr, T_new)
        pcs.append(pc_curr_global)
    
    return np.array(T_global)

def main():
    dataset = 20
    # 1) Load dataset
    (
        encoder_counts,      
        encoder_stamps,  
        lidar_angle_min,
        lidar_angle_max,
        lidar_angle_increment,
        lidar_range_min,
        lidar_range_max,
        lidar_ranges,
        lidar_stamps,
        imu_angular_velocity,
        imu_linear_acceleration,
        imu_stamps,
        disp_stamps,
        rgb_stamps, 
    ) = load_dataset(dataset=dataset)
   
    # 2) Compute robot trajectory
    FR = encoder_counts[0, :]
    FL = encoder_counts[1, :]
    RR = encoder_counts[2, :]
    RL = encoder_counts[3, :]
    yaw_rate = imu_angular_velocity[2, :]
    
    x, y, theta = compute_trajectory(FR, FL, RR, RL, yaw_rate, encoder_stamps, imu_stamps)
    theta = np.unwrap(theta)
    
    icp_T = compute_icp_trajectory(
        encoder_counts, encoder_stamps, imu_stamps, yaw_rate,
        lidar_ranges, lidar_stamps,
        lidar_angle_min, lidar_angle_increment, lidar_range_min, lidar_range_max
    )
    
    intrinsic = {
        'f_u': 585.05,
        'f_v': 585.05,
        'c_u': 242.94,
        'c_v': 315.84
    }
    # Robot-to-camera transform
    translation = np.array([0.18, 0.005, 0.36])
    rotation_angles = (0.0, -0.36, 0.021)  # roll, pitch, yaw
    T_robot_cam = get_transformation_matrix(translation, rotation_angles)
    
    # 3) Configure map for occupancy + texture
    MAP = {}
    MAP['res'] = np.array([0.05, 0.05])    # meters
    MAP['min'] = np.array([-20.0, -20.0])  # meters
    MAP['max'] = np.array([20.0, 20.0])    # meters
    MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
    isEven = MAP['size'] % 2 == 0
    MAP['size'][isEven] = MAP['size'][isEven] + 1  # Ensure that the map has an odd size so that the origin is in the center cell
    
    # 4) Accumulate floor points and colors from Kinect frames
    all_points_world = []
    all_colors = []
    icp_x_array = np.array([T[0, 2] for T in icp_T])
    icp_y_array = np.array([T[1, 2] for T in icp_T])
    icp_theta_array = np.array([np.arctan2(T[1, 0], T[0, 0]) for T in icp_T])
    
    for i in range(len(disp_stamps)):
        t_kinect = disp_stamps[i]
        # print(f"Processing frame {i+1}/{len(disp_stamps)} at time {t_kinect:.2f}")
        
        # Interpolate x, y, theta at time t_kinect:
        x_pose = np.interp(t_kinect, lidar_stamps, icp_x_array)
        y_pose = np.interp(t_kinect, lidar_stamps, icp_y_array)
        th_pose = np.interp(t_kinect, lidar_stamps, icp_theta_array)
        
        T_world_robot = build_2D_transform(x_pose, y_pose, th_pose)
        
        disp_img_path = f"/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR2/dataRGBD/Disparity{dataset}/disparity{dataset}_{i+1}.png"
        rgb_img_path = f"/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR2/dataRGBD/RGB{dataset}/rgb{dataset}_{i+1}.png"
        
        disp_img = cv2.imread(disp_img_path, cv2.IMREAD_UNCHANGED)  
        rgb_img  = cv2.imread(rgb_img_path)      # BGR by default
        if disp_img is None or rgb_img is None:  
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        h_disp, w_disp = disp_img.shape[:2]
        h_rgb,  w_rgb  = rgb_img.shape[:2]

        # Downsample if desired, to save time.
        # For example, step = 2
        # import pdb
        # pdb.set_trace()
        step = 6

        for ii in range(0, h_disp, step):
            if ii % 50 == 0:
                print(f"  scanning row {ii}/{h_disp}")
            for jj in range(0, w_disp, step):
                d = disp_img[ii, jj]
                # Sometimes disparity can be 0 or invalid; skip these pixels
                if d <= 0:
                    continue

                depth = compute_depth_from_disparity(d)
                # Filter out invalid or extremely large depths if needed
                if depth < 0.1 or depth > 5.0:
                    continue

                # Map (ii, jj) in disparity to (rgbi, rgbj) in the RGB image
                rgbi, rgbj = rgb_index_from_disparity(ii, jj, 
                                                        -0.00304 * d + 3.31, 
                                                        h_rgb, w_rgb)
                # Skip if out of bounds
                if (rgbi < 0 or rgbi >= h_rgb or 
                    rgbj < 0 or rgbj >= w_rgb):
                    continue

                # (d) Get the color from the RGB image
                color = rgb_img[rgbi, rgbj]  # [R, G, B]

                # (e) Convert the pixel to a 3D camera point and then to world coordinates
                xyz_cam = compute_camera_point(ii, jj, depth, intrinsic)
                xyz_world = get_world_point(xyz_cam, T_robot_cam, T_world_robot)

                # Store for later texturing
                all_points_world.append(xyz_world)
                all_colors.append(color)
                
    # 5) Build the final floor color map
    all_points_world = np.array(all_points_world)
    all_colors = np.array(all_colors)
    color_map = build_floor_color_map(all_points_world, all_colors, MAP, z_threshold=0.3)
    
    grid_x = ((icp_x_array - MAP['min'][0]) / MAP['res'][0]).astype(int)
    grid_y = ((icp_y_array - MAP['min'][1]) / MAP['res'][1]).astype(int)
    grid_y = color_map.shape[0] - grid_y  
    
    data = np.load("/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR2/data/occupancy_20.npz")
    occ_map = data['map']
    occ_gray = np.stack([occ_map]*3, axis=-1)
    threshold = 0.3
    composite_img = np.where(occ_map[..., None] < threshold, color_map, occ_gray)

    # 6) Visualize or save the texture map
    plt.figure(figsize=(8,8))
    # Note: color_map is indexed as color_map[x_index, y_index]
    plt.imshow(color_map.transpose(1, 0, 2)[::-1, :, :])
    plt.plot(grid_x, grid_y, 'r-', linewidth=2)
    plt.title("Floor Texture Map")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.imshow(composite_img)
    plt.title("Overlay Texture on Occupancy Map")
    plt.axis("off")
    plt.show()
    # Uncomment below to visualize the point cloud in 3D
    # pcd_all = o3d.geometry.PointCloud()
    # pcd_all.points = o3d.utility.Vector3dVector(all_points_world)
    # pcd_all.colors = o3d.utility.Vector3dVector(all_colors / 255.0)
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # o3d.visualization.draw_geometries([pcd_all, origin])

if __name__ == "__main__":
    main()


        
    
