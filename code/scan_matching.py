import numpy as np
import matplotlib.pyplot as plt
from load_data import load_dataset
from scipy.spatial import cKDTree
from part1 import compute_trajectory



def Kabsch (source_pc,target_pc):
    centroid_source_points = np.mean(source_pc,axis = 0)
    centroid_target_points = np.mean(target_pc,axis = 0)
    
    source_points_center = source_pc - centroid_source_points
    target_points_center = target_pc  - centroid_target_points
    
    Q = source_points_center.T @ target_points_center
    U,S,Vt = np.linalg.svd(Q)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    P= centroid_target_points - R @ centroid_source_points
    
    return R, P

def transform_points(points, R, P):
    """
    The expression (R @ points.T).T rotates the points. Here, points is an N×3 array (with N points in 3D).
    Transposing points with .T changes its shape to 3×N Multiplying with the 3×3 rotation matrix R rotates all points simultaneously.
    Transposing back with .T returns the rotated points in shape N×3.
    """
    return (R @ points.T).T + P
  
def ICP(source_points,target_points,init_odometry,max_iterations=20,tolerance=1e-7):
    T=init_odometry.copy()
    kdtree = cKDTree(target_points)
    current_source = source_points.copy()
    prev_error = float('inf')

    
    for iteration in range(max_iterations):
        # (Step 1) Data Association: find closet points
        dist, indices = kdtree.query(current_source)
        corresponding_targets = target_points[indices]
        #(Step 2) Apply Kabsch algorithm
        R, P = Kabsch(current_source, corresponding_targets)
        #(Step 3)Updating the Source Points
        current_source = transform_points(current_source, R, P)
        T_update = np.eye(4)
        T_update[:3, :3] = R   
        T_update[:3, 3]  = P
        T = T_update @ T
        mse = np.mean(dist ** 2) 
        if abs(prev_error - mse) < tolerance:
            break
        prev_error = mse
    return T

def lidar_scan_to_pc(lidar_scan, angle_min, angle_increment, lidar_range_min, lidar_range_max):
    points = []
    angle_increment = float(angle_increment[0, 0])
    for i, r in enumerate(lidar_scan):
        if r < lidar_range_min or r > lidar_range_max:
            continue
        angle = angle_min + i * angle_increment
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        points.append([x, y, 0])
    return np.array(points)

def odom_relative_transform(x1, y1, theta1, x2, y2, theta2):
    dx = x2 - x1
    dy = y2 - y1
    dtheta = theta2 - theta1

    T = np.eye(4)
    
    """""
    [ cos(dtheta)  -sin(dtheta)   0     dx]
    [ sin(dtheta)   cos(dtheta)   0     dy]
    [     0             0         1      0]
    [     0             0         0      1]
    
    """""
    
    T[0, 0] = np.cos(dtheta)
    T[0, 1] = -np.sin(dtheta)
    T[1, 0] = np.sin(dtheta)
    T[1, 1] = np.cos(dtheta)
    T[0, 3] = dx
    T[1, 3] = dy
    
    return T

def apply_4x4_transform(points, T):
    
    N = points.shape[0]
    homog = np.hstack([points, np.ones((N, 1))])  # Nx4
    transformed = (T @ homog.T).T  #T@homog -> 4xN .T-> Nx4
    return transformed[:, :3]  #only take x,y,z

def get_pose_at_timestamp(t_query, odom_stamps, x_arr, y_arr, theta_arr):
    
    x_intp = np.interp(t_query, odom_stamps, x_arr)
    y_intp = np.interp(t_query, odom_stamps, y_arr)
    theta_intp = np.interp(t_query, odom_stamps, theta_arr)  
    return x_intp, y_intp, theta_intp

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
    FR = encoder_counts[0,:]
    FL = encoder_counts[1,:]
    RR = encoder_counts[2,:]
    RL = encoder_counts[3,:]
    yaw_rate = imu_angular_velocity[2,:]
    
    """""
    x_shape(4956,) y_shape(4956,) theta_shape(4956,)
    """""
    x, y, theta = compute_trajectory(FR, FL, RR, RL,yaw_rate, encoder_stamps, imu_stamps) 
    theta = np.unwrap(theta)
    
    n_scans = lidar_ranges.shape[1] #shape(4962,)
    scan0 = lidar_ranges.T[0] #shape(1081,)
    pc0 = lidar_scan_to_pc(scan0, lidar_angle_min, lidar_angle_increment, lidar_range_min, lidar_range_max)  # Convert first LiDAR scan to point cloud
    
    pcs = []
    pcs.append(pc0)
    T = [np.eye(4)] #Identity matrix as initial transformation

    for i in range(n_scans - 1):
        import pdb
        pdb.set_trace()
        scan_next = lidar_ranges.T[i+1]
        pc_next = lidar_scan_to_pc(scan_next, lidar_angle_min, lidar_angle_increment,
                                   lidar_range_min, lidar_range_max)
        # Time alignment: get corresponding odometry pose using LiDAR timestamps
        t_lidar_i   = lidar_stamps[i]
        t_lidar_i1  = lidar_stamps[i+1]
        x_i, y_i, th_i    = get_pose_at_timestamp(t_lidar_i,  encoder_stamps, x, y, theta)
        x_i1, y_i1, th_i1 = get_pose_at_timestamp(t_lidar_i1, encoder_stamps, x, y, theta)
        
        #Compute relative transformation T_odom from current to next pose
        T_odom = odom_relative_transform(x_i, y_i, th_i, x_i1, y_i1, th_i1)
        T_init = T[-1] @ T_odom
        pc_next_init = apply_4x4_transform(pc_next, T_init)
        target_points = pcs[-1]
        
        T_icp = ICP(pc_next_init, target_points, init_odometry=T_init,
                max_iterations=20, tolerance=1e-7)
        T_new = T_icp @ T_init
        T.append(T_new)
        
        pc_next_global = apply_4x4_transform(pc_next, T_new)
        pcs.append(pc_next_global)
        
    plt.figure(figsize=(8,6))
    
    plt.plot(x, y, 'b-', linewidth=2,label='Odometry (Encoders + IMU Yaw)')
    icp_x = [T_i[0, 3] for T_i in T]
    icp_y = [T_i[1, 3] for T_i in T]


    plt.plot(icp_x, icp_y, 'g-',linewidth=2, label='ICP Corrected Trajectory')

    for pc_g in pcs:
        plt.scatter(pc_g[:, 0], pc_g[:, 1], s=1, c='k')
    
    plt.title("Robot Trajectory + Aligned Lidar Scans")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()
     
if __name__ == '__main__':  
    main()
