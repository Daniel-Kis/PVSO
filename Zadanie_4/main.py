
import numpy as np
import open3d as o3d
import os
import sys
# sys.path.append('')
# import open3d_tutorial as o3dtud

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)
    # Showing outliers (red) and inliers (gray)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return ()

def main():
    # dano location "C:\\Users\\Dano\\PycharmProjects\\Zadanie4\\my_pts.ply"
    # martin location "C:\\Users\\Lenovo\\PycharmProjects\\PVSO_zad1\\Zadanie_4\\my_pts.ply"
    cloud_kinect = o3d.io.read_point_cloud("C:\\Users\\Lenovo\\PycharmProjects\\PVSO_zad1\\Zadanie_4\\my_pts.ply")
    cloud_downloaded = o3d.io.read_point_cloud("C:\\Users\\Lenovo\\PycharmProjects\\PVSO_zad1\\Zadanie_4\\hmmPLY.ply")
    o3d.visualization.draw_geometries([cloud_kinect])
    o3d.visualization.draw_geometries([cloud_downloaded])

    # picture1 = o3d.io.read_point_cloud("C:\\Users\\Lenovo\\PycharmProjects\\PVSO_zad1\\Zadanie_4\\output_big2.pcd")
    # o3d.visualization.draw_geometries([picture1])

    # # Downsample the point cloud with a voxel of 0.05
    # # for my_pts.ply
    # voxel_down_cloud_kinect = cloud_kinect.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel_down_cloud_kinect])
    #
    # # for hmmPLY.ply
    # voxel_down_cloud_downloaded = cloud_downloaded.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel_down_cloud_downloaded])
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
    # # for my_pts.ply
    # cloud_kinect_removed, ind = cloud_kinect.remove_radius_outlier(nb_points=20, radius=0.05)
    # # display_inlier_outlier(cloud_kinect, ind)
    # o3d.visualization.draw_geometries([cloud_kinect_removed])

    # for hmmPLY.ply
    cloud_downloaded_removed, ind = cloud_downloaded.remove_radius_outlier(nb_points=30, radius=0.05)
    # display_inlier_outlier(cloud_downloaded, ind)
    o3d.visualization.draw_geometries([cloud_downloaded_removed])


    o3d.visualization.draw_geometries()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
