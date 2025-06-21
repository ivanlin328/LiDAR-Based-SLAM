import numpy as np
import cv2
import open3d as o3d

# generate sample pc from disparity images

# IMREAD_UNCHANGED ensures we preserve the precision on depth
disp_img = cv2.imread("/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR2/dataRGBD/Disparity20/disparity20_1.png", cv2.IMREAD_UNCHANGED)

# note that cv2 imports as bgr, so colors may be wrong.
bgr_img = cv2.imread("/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR2/dataRGBD/RGB20/rgb20_1.png")
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# from writeup, compute correspondence
height, width = disp_img.shape

dd = np.array(-0.00304 * disp_img + 3.31)
depth = 1.03 / dd

mesh = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')  
i_idxs = mesh[0].flatten()
j_idxs = mesh[1].flatten()

rgb_i = np.array((526.37 * i_idxs + 19276 - 7877.07 * dd.flatten()) / 585.051, dtype=np.int32)  # force int for indexing
rgb_j = np.array((526.37 * j_idxs + 16662) / 585.051, dtype=np.int32)

# some may be out of bounds, just clip them
rgb_i = np.clip(rgb_i, 0, height - 1)
rgb_j = np.clip(rgb_j, 0, width - 1)

colors = rgb_img[rgb_i, rgb_j]

# lets visualize the image using our transformation to make sure things look correct (using bgr for opencv)
bgr_colors = bgr_img[rgb_i, rgb_j]
cv2.imshow("color", bgr_colors.reshape((height, width, 3)))

uv1 = np.vstack([j_idxs, i_idxs, np.ones_like(i_idxs)])
K = np.array([[585.05, 0, 242.94],
              [0, 585.05, 315.84],
              [0, 0, 1]])

# project images to 3d points
points = depth.flatten() * (np.linalg.inv(K) @ uv1)

oRr = np.array([[0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]])
# we want rRo because we have points in optical frame and want to move them to the regular frame.
points = oRr.T @ points

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.T)
pcd.colors = o3d.utility.Vector3dVector(colors / 255)  # open3d expects color channels 0-1, opencv uses uint8 0-255

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.5)  # visualize the camera regular frame for reference.

o3d.visualization.draw_geometries([pcd, origin])  # display the pointcloud and origin