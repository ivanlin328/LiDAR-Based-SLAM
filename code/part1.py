import numpy as np
import matplotlib.pyplot as plt
from load_data import load_dataset

def compute_trajectory(FR, FL, RR, RL,yaw_rate, encoder_stamps, imu_stamps):
    n = len(encoder_stamps)
     
    # Initialize trajectory arrays
    x = np.zeros(n)
    y = np.zeros(n)
    theta = np.zeros(n)
    # Initial pose is set to (0, 0, 0)
    x[0],y[0],theta[0] = 0.0, 0.0, 0.0
    for i in range(1,n):
        dt = encoder_stamps[i] - encoder_stamps[i-1]
        if dt <= 0:
            # In case of any timestamp issues, set a small dt
            dt = 1e-3
        right_wheels_dist = 0.0022*(FR[i]+RR[i])/2 
        left_wheels_dist = 0.0022*(FL[i]+RL[i])/2 
        v = (right_wheels_dist + left_wheels_dist) / 2.0 / dt
    
    # Using np.interp to linearly interpolate the yaw rate
        current_time = encoder_stamps[i]
        omega = np.interp(current_time, imu_stamps, yaw_rate)
        
        theta[i] = theta[i-1] + omega * dt 
        x[i] = x[i-1] + v * np.cos(theta[i-1]) * dt
        y[i] = y[i-1] + v * np.sin(theta[i-1]) * dt
        
    return x,y,theta

    
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
    ) = load_dataset(dataset=21)
    FR = encoder_counts[0,:]
    FL = encoder_counts[1,:]
    RR = encoder_counts[2,:]
    RL = encoder_counts[3,:]
    yaw_rate = imu_angular_velocity[2,:]
    
    x,y,theta = compute_trajectory(FR, FL, RR, RL,yaw_rate, encoder_stamps, imu_stamps)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y,label='Odometry (Encoders + IMU Yaw)')
    plt.axis('equal')
    plt.title('Differential-Drive Robot trajectory for DataSet 20')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
 
    
if __name__ == '__main__':
    main()