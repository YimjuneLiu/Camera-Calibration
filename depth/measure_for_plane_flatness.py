import open3d as o3d
import numpy as np
import math
import matplotlib.pyplot as plt

ply = o3d.io.read_point_cloud("./Pointcloud_NEURAL.ply")
print(ply)
# print(pcd)

# 
distance_threshold = 0.05   # 5 cm
ransac_n = 3
num_iterations = 1000

plane_model, inliers = ply.segment_plane(distance_threshold, ransac_n, num_iterations)

# 
[a, b, c, d] = plane_model

inlier_cloud = ply.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 0, 1.0])
print(inlier_cloud)

tree = o3d.geometry.KDTreeFlann(inlier_cloud)
inlier_cloud.colors[1500] = [0.5, 0, 0.5]
[num_k, idx_k, _] = tree.search_knn_vector_3d(inlier_cloud.points[500000], 100000)
np.asarray(inlier_cloud.colors)[idx_k[1:], :] = [1, 0, 0]

# bounding_ploy = np.array([[1.3, 6.1, 0],
#                           [0.864, -6.518, 0],
#                           [15, -41, 0],
#                           [42.774, -41.181, 0]
#                           ], dtype=np.float32).reshape([-1, 3]).astype("float64")
# center_point = np.asarray(inlier_cloud.points[100000])
center_point = inlier_cloud.points[100000]
print("center point: x{}y{}z{}".format(center_point[0], center_point[1], center_point[2]))
aera_point = np.asarray(inlier_cloud.points)[idx_k[1:], :]
heights = aera_point[:, 2]
# print(heights)

index = []
height = []
for i in range(len(heights)):
    index.append(i)
    height.append(heights[i]*100)

plt.bar(index, height)
plt.ylim(-140,-150)
plt.title("points' height")
plt.show()

# o3d.visualization.draw_geometries([inlier_cloud])
