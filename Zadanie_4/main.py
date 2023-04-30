
import numpy as np
import open3d as o3d
import os
import sys
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# def display_inlier_outlier(cloud, ind):
#     inlier_cloud = cloud.select_down_sample(ind)
#     outlier_cloud = cloud.select_down_sample(ind, invert=True)
#     # Showing outliers (red) and inliers (gray)
#     outlier_cloud.paint_uniform_color([1, 0, 0])
#     inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
#     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
#     return ()


def main():
    # dano location "C:\\Users\\Dano\\PycharmProjects\\Zadanie4\\my_pts.ply"
    # martin location "C:\\Users\\Lenovo\\PycharmProjects\\PVSO_zad1\\Zadanie_4\\my_pts.ply"
    cloud_kinect = o3d.io.read_point_cloud("C:\\Users\\Dano\\PycharmProjects\\Zadanie4\\my_pts.ply")
    cloud_downloaded = o3d.io.read_point_cloud("C:\\Users\\Dano\\PycharmProjects\\Zadanie4\\hmmPLY.ply")
    o3d.visualization.draw_geometries([cloud_kinect])
    o3d.visualization.draw_geometries([cloud_downloaded])

    # picture1 = o3d.io.read_point_cloud("C:\\Users\\Lenovo\\PycharmProjects\\PVSO_zad1\\Zadanie_4\\output_big2.pcd")
    # o3d.visualization.draw_geometries([picture1])

    # Downsample the point cloud with a voxel of 0.05
    # for my_pts.ply
    voxel_down_cloud_kinect = cloud_kinect.voxel_down_sample(voxel_size=0.03)
    o3d.visualization.draw_geometries([voxel_down_cloud_kinect])

    # for hmmPLY.ply
    voxel_down_cloud_downloaded = cloud_downloaded.voxel_down_sample(voxel_size=0.03)
    o3d.visualization.draw_geometries([voxel_down_cloud_downloaded])
    #
    #
    # # Every 15th points are selected
    # # for my_pts.ply
    # uni_down_cloud_kinect = cloud_kinect.uniform_down_sample(every_k_points=15)
    # o3d.visualization.draw_geometries([uni_down_cloud_kinect])
    #
    # # for hmmPLY.ply
    # uni_down_cloud_downloaded = cloud_downloaded.uniform_down_sample(every_k_points=15)
    # o3d.visualization.draw_geometries([uni_down_cloud_downloaded])

    # radius removal of points
    # for my_pts.ply
    cloud_kinect_removed, ind = voxel_down_cloud_kinect.remove_radius_outlier(nb_points=5, radius=0.05)
    # display_inlier_outlier(cloud_kinect, ind)
    o3d.visualization.draw_geometries([cloud_kinect_removed])

    # for hmmPLY.ply
    cloud_downloaded_removed, ind = voxel_down_cloud_downloaded.remove_radius_outlier(nb_points=5, radius=0.05)
    # display_inlier_outlier(cloud_downloaded, ind)
    o3d.visualization.draw_geometries([cloud_downloaded_removed])

    # Birch
    model = Birch(branching_factor=50, n_clusters=5, threshold=0.5)
    data = np.asarray(cloud_downloaded_removed.points)
    model.fit(data)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    pred = model.predict(data)
    colors = np.zeros([pred.size, 3], dtype=float)
    for i in range(pred.size):
        if pred[i] == 0:
            colors[i, :] = [0, 0, 255]
        elif pred[i] == 1:
            colors[i, :] = [0, 255, 0]
        elif pred[i] == 3:
            colors[i, :] = [255, 0, 255]
        elif pred[i] == 4:
            colors[i, :] = [255, 0, 0]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

    # k-menas
    K = 5  # number of clusters
    model2 = KMeans(K)
    data2 = np.asarray(cloud_downloaded_removed.points)
    model2.fit(data2)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data2)

    pred = model2.predict(data2)
    colors = np.zeros([pred.size, 3], dtype=float)
    for i in range(pred.size):
        if pred[i] == 0:
            colors[i, :] = [0, 0, 255]
        elif pred[i] == 1:
            colors[i, :] = [0, 255, 0]
        elif pred[i] == 3:
            colors[i, :] = [255, 0, 255]
        elif pred[i] == 4:
            colors[i, :] = [255, 0, 0]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()

