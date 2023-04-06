import sys

# sys.path.insert(0, '$Home\\libfreenect2')

import numpy as np
import open3d as o3d
import cv2 as cv2
# from freenect2 import Device, FrameType
# import pandas as pd
# from pyntcloud import PyntCloud

# from pylibfreenect2 import Freenect2, SyncMultiFrameListener
# from pylibfreenect2 import FrameType, Registration, Frame
# from pylibfreenect2 import createConsoleLogger, setGlobalLogger
# from pylibfreenect2 import LoggerLevel



def get2DPicture():
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        try:
            from pylibfreenect2 import OpenCLPacketPipeline
            pipeline = OpenCLPacketPipeline()
        except:
            from pylibfreenect2 import CpuPacketPipeline
            pipeline = CpuPacketPipeline()
    print("Packet pipeline:", type(pipeline).__name__)

    # Create and set logger
    logger = createConsoleLogger(LoggerLevel.Debug)
    setGlobalLogger(logger)

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)

    listener = SyncMultiFrameListener(
        FrameType.Color | FrameType.Ir | FrameType.Depth)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()

    # NOTE: must be called after device.start()
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    # Optinal parameters for registration
    # set True if you need
    need_bigdepth = False
    need_color_depth_map = False

    bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
    color_depth_map = np.zeros((424, 512), np.int32).ravel() \
        if need_color_depth_map else None

    while True:
        frames = listener.waitForNewFrame()

        color = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered,
                           bigdepth=bigdepth,
                           color_depth_map=color_depth_map)

        # NOTE for visualization:
        # cv2.imshow without OpenGL backend seems to be quite slow to draw all
        # things below. Try commenting out some imshow if you don't have a fast
        # visualization backend.
        cv2.imshow("ir", ir.asarray() / 65535.)
        cv2.imshow("depth", depth.asarray() / 4500.)
        cv2.imshow("color", cv2.resize(color.asarray(),
                                       (int(1920 / 3), int(1080 / 3))))
        cv2.imshow("registered", registered.asarray(np.uint8))

        if need_bigdepth:
            cv2.imshow("bigdepth", cv2.resize(bigdepth.asarray(np.float32),
                                              (int(1920 / 3), int(1082 / 3))))
        if need_color_depth_map:
            cv2.imshow("color_depth_map", color_depth_map.reshape(424, 512))

        listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

    device.stop()
    device.close()

    sys.exit(0)


def callKinect():
    # Open the default device and capture a color and depth frame.
    device = Device()
    frames = {}
    with device.running():
        for type_, frame in device:
            frames[type_] = frame
            if FrameType.Color in frames and FrameType.Depth in frames:
                break

    # Use the factory calibration to undistort the depth frame and register the RGB
    # frame onto it.
    rgb, depth = frames[FrameType.Color], frames[FrameType.Depth]
    undistorted, registered, big_depth = device.registration.apply(
        rgb, depth, with_big_depth=True)

    # Combine the depth and RGB data together into a single point cloud.
    with open('output2.pcd', 'wb') as fobj:
        device.registration.write_pcd(fobj, undistorted, registered)

    with open('output_big2.pcd', 'wb') as fobj:
        device.registration.write_big_pcd(fobj, big_depth, rgb)


def use_o3d(pts, write_text):
    pcd = o3d.geometry.PointCloud()

    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
    # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(pts)

    # print(np.asarray(pcd.vertex_colors))

    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)

    # read ply file
    # pcd = o3d.io.read_point_cloud('my_pts.ply')

    # visualize
    # o3d.visualization.draw_geometries([pcd])


def use_o3d2(pts, colors, write_text):
    pcd = o3d.geometry.PointCloud()

    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
    # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)

    # read ply file
    # pcd = o3d.io.read_point_cloud('my_pts.ply')

    # visualize
    # o3d.visualization.draw_geometries([pcd])


def use_pyntcloud(pts, write_text):

    n = len(pts)

    # The points must be written as a dataframe,
    # ref: https://stackoverflow.com/q/70304087/6064933
    data = {'x': pts[:, 0],
            'y': pts[:, 1],
            'z': pts[:, 2],
            'red': np.random.rand(n),
            'blue': np.random.rand(n),
            'green': np.random.rand(n)
            }

    # build a cloud
    cloud = PyntCloud(pd.DataFrame(data))

    # the argument for writing ply file can be found in
    # https://github.com/daavoo/pyntcloud/blob/7dcf5441c3b9cec5bbbfb0c71be32728d74666fe/pyntcloud/io/ply.py#L173
    cloud.to_file('my_pts2.ply', as_text=write_text)

def main():
    # callKinect()
    # #get2DPicture()
    file = o3d.io.read_point_cloud("C:\\Users\\Dano\\PycharmProjects\\PVSO_Zadanie3\\Kinect\\output_big2.pcd")
    # o3d.visualization.draw_geometries([file])
    print(np.asarray(file.points).shape)
    print(np.asarray(file.colors).shape)
    # file2 = PyntCloud.from_file("output2.pcd")
    file = np.asarray(file.points)
    print(np.asarray(file).shape)
    use_o3d(file, True)
    # pts = np.asarray(file.points)
    # colors = np.asarray(file.colors)
    # use_o3d2(pts, colors, True)
    cloud = o3d.io.read_point_cloud("C:\\Users\\Dano\\PycharmProjects\\PVSO_Zadanie3\\Kinect\\my_pts.ply")
    print(np.asarray(cloud.points).shape)
    print(np.asarray(cloud.colors).shape)
    # cloud = np.asarray(cloud.points, dtype=np.uint8)
    o3d.visualization.draw_geometries([cloud])
    #
    # use_pyntcloud(file, False)
    #
    # ply = PyntCloud.from_file("my_pts2.ply")
    # ply.plot(mesh=True, backend="threejs")

if __name__ == '__main__':
    main()

