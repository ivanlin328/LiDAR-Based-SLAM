import numpy as np
import matplotlib.pyplot as plt
from load_data import load_dataset
from scipy.spatial import cKDTree
from part1 import compute_trajectory

def Kabsch(source_pc, target_pc):
    centroid_source = np.mean(source_pc, axis=0)
    centroid_target = np.mean(target_pc, axis=0)
    src_centered = source_pc - centroid_source
    tgt_centered = target_pc - centroid_target
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_target - R @ centroid_source
    return R, t

def transform_points(points, R, t):
    return (R @ points.T).T + t

def ICP(source_points, target_points, init_odometry, max_iterations=20, tolerance=1e-7, distance_threshold=0.1):
    # Here, init_odometry is used as the initial global pose,
    # and since source_points has already been pre-transformed,
    # the ICP initial estimate is set to identity (only solving for the additional correction).
    T_total = np.eye(3)
    # Copy source_points to work on as 2D points
    current_source = source_points.copy()
    prev_error = float('inf')
    tree = cKDTree(target_points)
    
    for i in range(max_iterations):
        distances, indices = tree.query(current_source)
        mask = distances < distance_threshold
        if np.sum(mask) < 3:
            break
        src_inliers = current_source[mask]
        dst_inliers = target_points[indices[mask]]
        R_delta, t_delta = Kabsch(src_inliers, dst_inliers)
        # Update the ICP transformation
        T_delta = np.array([[R_delta[0,0], R_delta[0,1], t_delta[0]],
                            [R_delta[1,0], R_delta[1,1], t_delta[1]],
                            [0, 0, 1]])
        T_total = T_delta @ T_total
        # Update the source point cloud
        current_source = (T_delta @ np.hstack([current_source, np.ones((current_source.shape[0], 1))]).T).T[:, :2]
        error = np.mean(distances[mask])
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    return T_total

def lidar_scan_to_pc(lidar_scan, angle_min, angle_increment, lidar_range_min, lidar_range_max):
    points = []
    angle_inc = float(angle_increment[0, 0])
    for i, r in enumerate(lidar_scan):
        if r < lidar_range_min or r > lidar_range_max:
            continue
        angle = angle_min + i * angle_inc
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        points.append([x, y, 0])
    return np.array(points)

def odom_relative_transform(x1, y1, theta1, x2, y2, theta2):
    dx = x2 - x1
    dy = y2 - y1
    dtheta = theta2 - theta1
    T = np.eye(3)
    T[0,0] = np.cos(dtheta)
    T[0,1] = -np.sin(dtheta)
    T[1,0] = np.sin(dtheta)
    T[1,1] = np.cos(dtheta)
    T[0,2] = dx
    T[1,2] = dy
    return T

def apply_transform_2d(points, T):
    # Assume that points are in homogeneous coordinates (N,3) with z=0;
    # after transformation, only x and y are returned.
    N = points.shape[0]
    homog = np.hstack([points[:, :2], np.ones((N, 1))])
    transformed = (T @ homog.T).T
    return transformed[:, :2]

def get_pose_at_timestamp(t_query, odom_stamps, x_arr, y_arr, theta_arr):
    x_intp = np.interp(t_query, odom_stamps, x_arr)
    y_intp = np.interp(t_query, odom_stamps, y_arr)
    theta_intp = np.interp(t_query, odom_stamps, theta_arr)
    return x_intp, y_intp, theta_intp

def main():
    # Load data
    (encoder_counts, encoder_stamps, lidar_angle_min, lidar_angle_max,
     lidar_angle_increment, lidar_range_min, lidar_range_max, lidar_ranges,
     lidar_stamps, imu_angular_velocity, imu_linear_acceleration, imu_stamps,
     disp_stamps, rgb_stamps) = load_dataset(dataset=21)
    
    FR = encoder_counts[0, :]
    FL = encoder_counts[1, :]
    RR = encoder_counts[2, :]
    RL = encoder_counts[3, :]
    yaw_rate = imu_angular_velocity[2, :]
    
    x, y, theta = compute_trajectory(FR, FL, RR, RL, yaw_rate, encoder_stamps, imu_stamps)
    theta = np.unwrap(theta)
    
    n_scans = lidar_ranges.shape[1]
    scan0 = lidar_ranges.T[0]
    pc0 = lidar_scan_to_pc(scan0, lidar_angle_min, lidar_angle_increment, lidar_range_min, lidar_range_max)
    
    pcs = []
    pcs.append(pc0)
    T_global = [np.eye(3)]  # Global pose as a 3x3 homogeneous matrix (considering only x, y, theta)
    
    # Use odometry as the initial global pose.
    # For each frame, compute the local relative transformation from odometry,
    # and then accumulate to update the global pose.
    for i in range(n_scans - 1):
        # Get the LiDAR scans from the previous and current frames
        scan_prev = lidar_ranges.T[i]
        scan_curr = lidar_ranges.T[i+1]
        points_prev = lidar_scan_to_pc(scan_prev, lidar_angle_min, lidar_angle_increment, lidar_range_min, lidar_range_max)
        points_curr = lidar_scan_to_pc(scan_curr, lidar_angle_min, lidar_angle_increment, lidar_range_min, lidar_range_max)
        
        # Use LiDAR timestamps to interpolate the corresponding odometry poses
        t_prev = lidar_stamps[i]
        t_curr = lidar_stamps[i+1]
        x_prev, y_prev, th_prev = get_pose_at_timestamp(t_prev, encoder_stamps, x, y, theta)
        x_curr, y_curr, th_curr = get_pose_at_timestamp(t_curr, encoder_stamps, x, y, theta)
        
        # Compute the odometry relative transformation (local 2D transformation)
        T_odom = odom_relative_transform(x_prev, y_prev, th_prev, x_curr, y_curr, th_curr)
        
        # Update the initial global pose using the previous global pose
        T_init = T_global[-1] @ T_odom
        
        # Transform the current point cloud to global coordinates (taking only x, y) using T_init
        pc_curr_init = apply_transform_2d(points_curr, T_init)
        # The ICP target is the previous point cloud represented as raw 2D points (transformed using the previous global pose)
        target_points = apply_transform_2d(points_prev, T_global[-1])
        
        # Call ICP, note that the initial estimate is the identity matrix
        T_icp = ICP(pc_curr_init, target_points, init_odometry=np.eye(3),
                    max_iterations=20, tolerance=1e-7, distance_threshold=0.1)
        
        # Update the global pose: first use T_init as the initial estimate, then apply the ICP correction
        T_new = T_icp @ T_init
        T_global.append(T_new)
        
        # Update the global point cloud (optional, for visualization)
        pc_curr_global = apply_transform_2d(points_curr, T_new)
        pcs.append(np.hstack([pc_curr_global, np.zeros((pc_curr_global.shape[0], 1))]))  # Append z=0
        
    # Plot the trajectory
    plt.figure(figsize=(8,6))
    plt.plot(x, y, 'b-', linewidth=2, label='Odometry Trajectory')
    global_x = [T[0, 2] for T in T_global]
    global_y = [T[1, 2] for T in T_global]
    plt.plot(global_x, global_y, 'g-', linewidth=2, label='ICP Corrected Trajectory')
    for pc in pcs:
        plt.scatter(pc[:,0], pc[:,1], s=1, c='k')
    plt.title("Robot Trajectory + Aligned Lidar Scans")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()




