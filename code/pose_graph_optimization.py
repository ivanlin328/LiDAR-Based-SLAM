import numpy as np
import gtsam
from gtsam import Pose2, BetweenFactorPose2, Values, noiseModel, NonlinearFactorGraph, LevenbergMarquardtOptimizer
import matplotlib.pyplot as plt
from texture import compute_icp_trajectory, build_2D_transform  # Using the ICP estimation from texture.py
from load_data import load_dataset
from part1 import compute_trajectory

def convert_T_to_Pose2(T):
    """
    Convert a 3x3 homogeneous matrix T to a gtsam.Pose2 (x, y, theta)
    """
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return Pose2(x, y, theta)

def compute_relative_pose(T_i, T_j):
    """
    Compute the relative transformation from pose i to pose j and convert it to Pose2.
    T_i and T_j are 3x3 homogeneous matrices.
    """
    T_rel = np.linalg.inv(T_i) @ T_j
    return convert_T_to_Pose2(T_rel)

def detect_loop_closures(T_global, distance_threshold=2.0, min_index_diff=200):
    """
    Automatically detect loop closures based on the initial global poses:
      - Only consider loop candidates if the indices of two poses differ by at least min_index_diff.
      - If the Euclidean distance between two poses is less than distance_threshold, it is considered a loop closure.
    Returns a list where each item is (i, j, relative_pose).
    
    Suggested parameters:
      distance_threshold = 2.0 (adjust as needed)
      min_index_diff = 200 (adjust based on your data length and scene)
    """
    loop_closures = []
    num_poses = T_global.shape[0]
    
    for i in range(num_poses):
        # j starts from (i + min_index_diff) to ensure the index difference is >= min_index_diff
        for j in range(i + min_index_diff, num_poses):
            dx = T_global[j, 0, 2] - T_global[i, 0, 2]
            dy = T_global[j, 1, 2] - T_global[i, 1, 2]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < distance_threshold:
                # Generate a loop closure factor
                T_rel = np.linalg.inv(T_global[i]) @ T_global[j]
                # Convert T_rel to gtsam.Pose2
                x_rel = T_rel[0, 2]
                y_rel = T_rel[1, 2]
                theta_rel = np.arctan2(T_rel[1, 0], T_rel[0, 0])
                rel_pose = gtsam.Pose2(x_rel, y_rel, theta_rel)
                
                loop_closures.append((i, j, rel_pose))
                
    print(f"Number of loop closures detected: {len(loop_closures)}")
    for (i, j, rel_pose) in loop_closures:
        print(f"  Loop: Node {i} -> {j}, (dx={rel_pose.x():.2f}, dy={rel_pose.y():.2f}, dtheta={rel_pose.theta():.2f})")
    
    return loop_closures

def optimize_pose_graph(T_global, loop_closures=None):
    """
    Perform pose graph optimization using GTSAM:
      - T_global: (N, 3, 3) initial global pose sequence.
      - loop_closures: Additional loop closure constraints, if detected.
    """
    num_poses = T_global.shape[0]
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Add a prior factor to fix the first pose
    prior_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3]))
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior_model))
    
    # Adjust odometry noise: reduce confidence in the initial trajectory and increase relative weight for loop closures
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.15]))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
    
    # 1. Add odometry factors for consecutive poses
    for i in range(num_poses - 1):
        rel_pose = compute_relative_pose(T_global[i], T_global[i + 1])
        graph.add(gtsam.BetweenFactorPose2(i, i + 1, rel_pose, odom_noise))
    
    # 2. Add loop closure constraints at fixed intervals (e.g., every 10 poses)
    skip = 10
    for i in range(0, num_poses - skip, skip):
        rel_pose = compute_relative_pose(T_global[i], T_global[i + skip])
        graph.add(gtsam.BetweenFactorPose2(i, i + skip, rel_pose, loop_noise))
    
    # 3. If automatically detected loop closures exist, add them as well
    if loop_closures is not None:
        for (i, j, rel_pose) in loop_closures:
            print(f"Adding loop closure factor: Node {i} -> {j}")
            graph.add(gtsam.BetweenFactorPose2(i, j, rel_pose, loop_noise))
    
    # 4. Build the initial estimate, intentionally injecting a small bias for clearer optimization effects
    for i in range(num_poses):
        pose = convert_T_to_Pose2(T_global[i])
        if i > 0:
            # Inject bias to all poses after the first one
            pose = gtsam.Pose2(pose.x() + 0.5, pose.y() - 0.3, pose.theta() + 0.1)
        initial_estimate.insert(i, pose)
    
    # 5. Optimize using Levenberg-Marquardt
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    optimized_poses = []
    for i in range(num_poses):
        optimized_poses.append(result.atPose2(i))
    return optimized_poses

def plot_trajectories(T_global, optimized_poses):
    """
    Plot a comparison of the ICP-generated trajectory and the optimized trajectory.
    """
    icp_x = [T[0, 2] for T in T_global]
    icp_y = [T[1, 2] for T in T_global]
    opt_x = [pose.x() for pose in optimized_poses]
    opt_y = [pose.y() for pose in optimized_poses]
    # Note: The odometry trajectory is obtained using compute_trajectory from part1.

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Odometry Trajectory')
    plt.plot(opt_x, opt_y, 'r.-', label='Optimized Trajectory')
    plt.plot(icp_x, icp_y, 'b.-', label='ICP Trajectory')
    
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Trajectory Comparison: ICP vs Optimized')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load data (make sure to select the same dataset as before)
    dataset = 2
    data = load_dataset(dataset=dataset)
    encoder_counts, encoder_stamps, lidar_angle_min, lidar_angle_max, lidar_angle_increment, \
      lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamps, imu_angular_velocity, \
      imu_linear_acceleration, imu_stamps, disp_stamps, rgb_stamps = data

    yaw_rate = imu_angular_velocity[2, :]
    
    # Compute the initial global pose sequence T_global (shape: (N, 3, 3)) using the ICP algorithm
    T_global = compute_icp_trajectory(
        encoder_counts, encoder_stamps, imu_stamps, yaw_rate,
        lidar_ranges, lidar_stamps,
        lidar_angle_min, lidar_angle_increment, lidar_range_min, lidar_range_max
    )
    print("Number of initial ICP poses:", T_global.shape[0])
    
    # Automatically detect loop closures based on proximity
    loop_closures = detect_loop_closures(T_global, distance_threshold=5.0, min_index_diff=5)
    print("Number of loop closures detected:", len(loop_closures))
    
    # Perform pose graph optimization
    optimized_poses = optimize_pose_graph(T_global, loop_closures)
    plot_trajectories(T_global, optimized_poses)
    
    print("Optimized pose results:")
    for i, pose in enumerate(optimized_poses):
        print(f"Pose {i}: {pose}")

if __name__ == '__main__':
    main()




    











