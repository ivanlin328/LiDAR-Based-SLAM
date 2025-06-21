
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from scipy.spatial import cKDTree

def rotation_z(theta):
    """
    Given an angle theta (in radians), return the 3x3 rotation matrix about the z-axis.
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

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
def mse_error(Z,M):
    diff = Z- M
    return np.mean(np.sum(diff**2, axis=1))
  
def transform_points(points, R, P):
    """
    The expression (R @ points.T).T rotates the points. Here, points is an N×3 array (with N points in 3D).
    Transposing points with .T changes its shape to 3×N Multiplying with the 3×3 rotation matrix R rotates all points simultaneously.
    Transposing back with .T returns the rotated points in shape N×3.
    """
    return (R @ points.T).T + P

  
def ICP(source_points,target_points,max_iterations=15,tolerance=1e-6):
    T= np.eye(4)
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
  
def ICP_with_yaw_candidates(source_points, target_points, yaw_candidates, max_iterations=20, tolerance=1e-7):

  best_error = float('inf')
  best_T = None
  best_yaw = None

  for yaw in yaw_candidates:
      # Convert candidate yaw angle from degrees to radians
      theta = np.deg2rad(yaw)
      # Create initial rotation matrix for yaw (rotation about z-axis)
      R_init = rotation_z(theta)
      T_init = np.eye(4)
      T_init[:3, :3] = R_init

      # Apply initial rotation to the source points
      init_source_points = transform_points(source_points, R_init, np.zeros(3))
      
      # Run ICP from the rotated source points toward the target
      T_icp = ICP(init_source_points, target_points, max_iterations, tolerance)
      
      # Combine the initial transformation with the ICP result
      T_total = T_icp @ T_init
      R= T_total[:3, :3]
      P=T_total[:3, 3]
      # Compute error: apply T_total to the original source points and compute MSE with target points
      transformed_source = transform_points(source_points, R, P)
      kdtree = cKDTree(target_points)
      distances, _ = kdtree.query(transformed_source)
      error = np.mean(distances**2)

      if error < best_error:
          best_error = error
          best_T = T_total
          best_yaw = yaw

  return best_T, best_error, best_yaw
  

if __name__ == "__main__":
   obj_name = 'liq_container' # drill or liq_container
   num_pc = 4 # number of point clouds

   source_pc = read_canonical_model(obj_name)
   yaw_candidates = list(range(0, 361, 15))
   
   for i in range(num_pc):
     target_pc = load_pc(obj_name, i)
     
     best_T, best_error, best_yaw = ICP_with_yaw_candidates(
             source_pc, target_pc,
             yaw_candidates=yaw_candidates,
             max_iterations=15,
             tolerance=1e-6
         )
     
     print("Best yaw angle (degrees):", best_yaw)
     print("Best registration error:", best_error)

     # visualize the estimated result
     visualize_icp_result(source_pc, target_pc, best_T)

