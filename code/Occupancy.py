import numpy as np
from pr2_utils import *
import matplotlib.pyplot as plt
from load_data import load_dataset
from part1 import compute_trajectory
from scan_matching2 import (
    lidar_scan_to_pc,
    odom_relative_transform,   # Make sure this function returns a 3×3 matrix
    get_pose_at_timestamp,
    ICP,
    apply_transform_2d         # Only accepts a 3×3 matrix
)
from tqdm import tqdm


def update_map_with_scan(MAP, global_points, robot_pose, log_free, log_occu):
    """
    global_points: Global point cloud with shape Nx2
    robot_pose: The robot position as (x, y)
    """
    robot_cell = np.floor((robot_pose - MAP['min']) / MAP['res']).astype(int).flatten()
    for point in global_points:
        cell = np.floor((point - MAP['min']) / MAP['res']).astype(int).flatten()
        # Check boundaries
        if not (0 <= cell[0] < MAP['size'][0] and 0 <= cell[1] < MAP['size'][1]):
            continue
        # Connect a ray between the robot cell and the target cell
        ray = bresenham2D(int(robot_cell[0]), int(robot_cell[1]), int(cell[0]), int(cell[1])).astype(int)
        # Mark the first n-1 points of the ray as free
        for r in ray.T[:-1]:
            if 0 <= r[0] < MAP['size'][0] and 0 <= r[1] < MAP['size'][1]:
                MAP['log_odds'][r[0], r[1]] += log_free
        # Mark the last point of the ray as occupied
        final_cell = ray.T[-1]
        if 0 <= final_cell[0] < MAP['size'][0] and 0 <= final_cell[1] < MAP['size'][1]:
            MAP['log_odds'][final_cell[0], final_cell[1]] += log_occu

def main():
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
    ) = load_dataset(dataset=20)
    
    # (1) Use Encoder+IMU to obtain odometry
    FR = encoder_counts[0, :]
    FL = encoder_counts[1, :]
    RR = encoder_counts[2, :]
    RL = encoder_counts[3, :]
    yaw_rate = imu_angular_velocity[2, :]
    x, y, theta = compute_trajectory(FR, FL, RR, RL, yaw_rate, encoder_stamps, imu_stamps)
    theta = np.unwrap(theta)

    # (2) Initialize the map
    MAP = {}
    MAP['res'] = np.array([0.05, 0.05])      # Each grid cell is 0.05m
    MAP['min'] = np.array([-20.0, -20.0])    
    MAP['max'] = np.array([20.0, 20.0])
    MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
    isEven = MAP['size'] % 2 == 0
    MAP['size'][isEven] = MAP['size'][isEven] + 1  # Ensure the center cell is exactly in the middle
    MAP['map'] = np.zeros(MAP['size'])
    MAP['log_odds'] = np.zeros(MAP['size'])
    
    lo_occ = 0.85
    lo_free = -0.4
    
    # (3) Get the first LiDAR scan
    n_scans = lidar_ranges.shape[1]
    scan0 = lidar_ranges.T[0]
    pc0 = lidar_scan_to_pc(scan0, lidar_angle_min, lidar_angle_increment,
                           lidar_range_min, lidar_range_max)
    
    # Assume the first scan corresponds to the global origin (0,0) and update the map
    robot_world = np.array([0.0, 0.0])
    # apply_transform_2d requires a 3x3 matrix; here we use np.eye(3) for no transformation
    pc0_2d = apply_transform_2d(pc0, np.eye(3))
    update_map_with_scan(MAP, pc0_2d, robot_world, lo_free, lo_occ)
    
    # (4) Prepare to store global point clouds and global poses
    pcs = [pc0]            # The original point cloud of the first frame
    T_global = [np.eye(3)] # Global pose, 3x3 homogeneous matrix

    for i in tqdm(range(n_scans - 1), desc='Processing scans'):
        # Get previous and current scan
        scan_prev = lidar_ranges.T[i]
        scan_curr = lidar_ranges.T[i+1]
        points_prev = lidar_scan_to_pc(scan_prev, lidar_angle_min, lidar_angle_increment,
                                       lidar_range_min, lidar_range_max)
        points_curr = lidar_scan_to_pc(scan_curr, lidar_angle_min, lidar_angle_increment,
                                       lidar_range_min, lidar_range_max)
        
        # Find the corresponding odometry pose using LiDAR timestamps
        t_prev = lidar_stamps[i]
        t_curr = lidar_stamps[i+1]
        x_prev, y_prev, th_prev = get_pose_at_timestamp(t_prev,  encoder_stamps, x, y, theta)
        x_curr, y_curr, th_curr = get_pose_at_timestamp(t_curr, encoder_stamps, x, y, theta)
        
        # (a) Use odometry to compute the 3x3 relative pose T_odom
        T_odom = odom_relative_transform(x_prev, y_prev, th_prev, x_curr, y_curr, th_curr)
        
        # (b) Accumulate from the previous global pose T_global[-1] to obtain the initial pose T_init
        T_init = T_global[-1] @ T_odom
        
        # (c) Transform the current scan to global coordinates (2D) using T_init
        pc_curr_init = apply_transform_2d(points_curr, T_init)
        
        # (d) Prepare the ICP target: previous scan transformed to global coordinates (2D)
        pc_prev_global = apply_transform_2d(points_prev, T_global[-1])
        
        # (e) Call ICP with an initial estimate of identity (3x3)
        T_icp = ICP(pc_curr_init, pc_prev_global,
                    init_odometry=np.eye(3),
                    max_iterations=20, tolerance=1e-7, distance_threshold=0.1)
        
        # (f) Obtain the new global pose
        T_new = T_icp @ T_init
        T_global.append(T_new)
        
        # (g) Transform the current scan to global coordinates for subsequent map updating
        pc_curr_global = apply_transform_2d(points_curr, T_new)
        pcs.append(pc_curr_global)
        
        # (h) Take the robot's (x, y) position for map updating
        robot_pose = np.array([T_new[0, 2], T_new[1, 2]])
        update_map_with_scan(MAP, pc_curr_global, robot_pose, lo_free, lo_occ)
    
    # (5) Map the log_odds to probabilities in [0,1]
    log_odds_clipped = np.clip(MAP['log_odds'], -20, 20)
    MAP['map'] = np.exp(log_odds_clipped) / (1 + np.exp(log_odds_clipped))
    np.savez("/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR2/data/occupancy_20.npz", map = MAP['map'])
    # (6) Plotting
    plt.figure()
    plot_map(MAP['map'], cmap='binary')
    plt.title('Grid map')
    plt.show()

if __name__ == '__main__':
    main()
