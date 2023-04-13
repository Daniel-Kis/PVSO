
import numpy as np
import open3d as o3d

def main():
    cloud_kinect = o3d.io.read_point_cloud("C:\\Users\\Dano\\PycharmProjects\\Zadanie4\\my_pts.ply")
    cloud_downloaded = o3d.io.read_point_cloud("C:\\Users\\Dano\\PycharmProjects\\Zadanie4\\hmmPLY.ply")
    o3d.visualization.draw_geometries([cloud_kinect])
    o3d.visualization.draw_geometries([cloud_downloaded])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
