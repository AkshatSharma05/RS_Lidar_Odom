import open3d as o3d
import time
import os
import glob
import numpy as np
import threading
import pygame

pcd_dir = "./pcd"
seen = set()
last_file = None
prev_file = None  

XPOS = 0.0
YPOS = 0.0
frame_skip = 0
frame_count = 0

vis = o3d.visualization.Visualizer()

vis.create_window("LIDAR")
pcd = o3d.geometry.PointCloud()
vis.get_render_option().background_color = [0, 0, 0]  
added = False

transformation = np.eye(4) #identity matrix of size 4x4
odometry = np.eye(4)
trajectory = [] 

# print("Watching directory:", pcd_dir)

import pygame

def pygame_plotter(get_pos, stop_event):
    pygame.init()
    size = 600
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("odom")
    clock = pygame.time.Clock()
    white = (255, 255, 255)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    points = []

    world_min = -20
    world_max = 20

    def world_to_screen(x, y):
        # Map world coordinates to screen
        sx = int((x - world_min) / (world_max - world_min) * size)
        sy = int(size - (y - world_min) / (world_max - world_min) * size)
        return sx, sy

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                return

        xpos, ypos = get_pos()
        points.append(world_to_screen(xpos, ypos))

        screen.fill(white)
        if len(points) > 1:
            pygame.draw.lines(screen, blue, False, points, 2)
        for pt in points:
            pygame.draw.circle(screen, black, pt, 2)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

def remove(prev_file):
    if prev_file and prev_file != last_file and os.path.exists(prev_file):
        try:
            os.remove(prev_file)
            # print(f"Deleted: {prev_file}")
        except Exception as e:
            print(f"Error deleting {prev_file}: {e}")

def calc_traj(last_file, new_pcd, trajectory):
    global XPOS, YPOS, odometry
    if last_file is not None:
        last_pcd = o3d.io.read_point_cloud(last_file)
        last_pcd = last_pcd.voxel_down_sample(voxel_size=0.1)
        try:
            transformation, inlier_rmse3 = perform_icp_point_to_plane(last_pcd, new_pcd)
            if not np.allclose(transformation, np.eye(4), atol=1e-2): #allclose checks if two matrices are equal
                proposed = np.dot(odometry, transformation) #next pose estimate

                #odometry[:2,3] -> first two elements of last column

                if np.linalg.norm(proposed[:2, 3] - odometry[:2, 3]) > 0.5: #distance between x and y coords of the poses
                    return 
                odometry = np.dot(odometry, transformation) # = proposed
            trajectory.append(odometry[:3, 3].copy())
            XPOS, YPOS = odometry[0, 3], odometry[1, 3]
        except:
            return
    else: #for the very first frame
        trajectory.append(odometry[:3, 3].copy())
        XPOS, YPOS = odometry[0, 3], odometry[1, 3]
    

def perform_icp_point_to_plane(source, target):
    # to find surface normals -> needed for point to plane  
    o3d.geometry.PointCloud.estimate_normals(
        source,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)) #nearest neighbour search 
    #the points are scattered and dont define a surface -> KDTree fits a plane through the points to visualiza surface

    o3d.geometry.PointCloud.estimate_normals(
        target,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

    threshold = 1.5 
    trans_init = transformation

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return reg_p2p.transformation, reg_p2p.inlier_rmse

stop_event = threading.Event()

def get_pos():
    return XPOS, YPOS
pygame_thread = threading.Thread(target=pygame_plotter, args=(get_pos, stop_event))
pygame_thread.start()

try:
    while True:
        pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
        for file in pcd_files:
            if file not in seen:
                # print("New file:", file)
                seen.add(file)

                new_pcd = o3d.io.read_point_cloud(file)
                new_pcd = new_pcd.voxel_down_sample(voxel_size=0.1)

                if not added:
                    pcd.points = new_pcd.points
                    # print(new_pcd.points)
                    vis.add_geometry(pcd)
                    added = True
                else:
                    pcd.points = new_pcd.points

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                calc_traj(last_file, new_pcd, trajectory)
                threading.Thread(target=remove, args=(prev_file,), daemon=True).start()
                # print(f"Current Position: X={XPOS:.4f}, Y={YPOS:.4f}")

                prev_file = last_file
                last_file = file
            time.sleep(0.001)

        time.sleep(0.001)
except KeyboardInterrupt:
    print("Exiting.")
    vis.destroy_window()
