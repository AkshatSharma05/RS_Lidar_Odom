import open3d as o3d
import time
import numpy as np
import threading
import pygame
import matplotlib.pyplot as plt

class LidarOdometry:
    def __init__(self, pcd_dir="./pcd"):
        self.pcd_dir = pcd_dir
        self.last_file = None

        self.XPOS = 0.0
        self.YPOS = 0.0
        self.frame_skip = 0
        self.frame_count = 0

        self.local_map = None

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("LIDAR")
        self.pcd = o3d.geometry.PointCloud()
        self.vis.get_render_option().background_color = [0, 0, 0]  
        self.added = False

        self.global_map = o3d.geometry.PointCloud()
        # self.mapper = o3d.visualization.Visualizer()
        # self.mapper.create_window("Map")
        # self.mapper.get_render_option().background_color = [0, 0, 0]


        self.transformation = np.eye(4) #identity matrix of size 4x4
        self.odometry = np.eye(4)
        self.trajectory = [] 

        self.stop_event = threading.Event()
        self.pygame_thread = threading.Thread(target=self.pygame_plotter, args=(self.get_pos, self.get_local_map, self.stop_event))
        self.pygame_thread.start()


    def get_pos(self):
        return self.XPOS, self.YPOS

    def pygame_plotter(self, get_pos, get_map, stop_event):
        pygame.init()
        size = 600
        screen = pygame.display.set_mode((size * 2, size))  # Two panels side by side
        pygame.display.set_caption("Odometry + 2D Lidar Map")
        clock = pygame.time.Clock()

        white = (255, 255, 255)
        blue = (0, 0, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)

        points = []

        world_min = -10
        world_max = 10

        def world_to_screen(x, y, offset_x=0):
            sx = int((x - world_min) / (world_max - world_min) * size) + offset_x
            sy = int(size - (y - world_min) / (world_max - world_min) * size)
            return sx, sy

        while not stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()
                    return

            screen.fill(white)

            # === ODOMETRY TRAJECTORY PANEL (Left half) ===
            xpos, ypos = get_pos()
            points.append(world_to_screen(xpos, ypos))
            if len(points) > 1:
                pygame.draw.lines(screen, blue, False, points, 2)
            for pt in points:
                pygame.draw.circle(screen, black, pt, 2)

            # === 2D LIDAR MAP PANEL (Right half) ===
            local_map = get_map()
            if local_map is not None:
                for pt in local_map:
                    x, y = pt[0], pt[1]
                    sx, sy = world_to_screen(x, y, offset_x=size)  # right panel
                    if 0 <= sx < size * 2 and 0 <= sy < size:
                        pygame.draw.circle(screen, red, (sx, sy), 1)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


    def get_local_map(self):
        return self.local_map

   
    def calc_traj(self, last_file, new_pcd, trajectory):
        def get_map_2d(new_pcd, z_min=-1.5, z_max=0):
            arr = np.asarray(new_pcd.points)
            mask = (arr[:, 2] >= z_min) & (arr[:, 2] <= z_max)
            arr_2d = arr[mask][:, :2]
            return arr_2d
        
        if last_file is not None:
            last_pcd = o3d.io.read_point_cloud(last_file)
            last_pcd = last_pcd.voxel_down_sample(voxel_size=0.1)
            try:
                self.local_map = get_map_2d(new_pcd, z_min=0.0, z_max=2.0)
                
                transformation, _ = self.perform_icp_point_to_plane(last_pcd, new_pcd)
                if not np.allclose(transformation, np.eye(4), atol=1e-2): #allclose checks if two matrices are equal
                    self.odometry = np.dot(self.odometry, transformation) # = proposed
            
                # kdtree = o3d.geometry.KDTreeFlann(self.global_map)
                # unique_points = []
                # for pt in np.asarray(new_pcd.points):
                #     [_, idx, _] = kdtree.search_radius_vector_3d(pt, 0.3)
                #     if len(idx) == 0:
                #         unique_points.append(pt)
                # if unique_points:
                #     self.global_map.points.extend(o3d.utility.Vector3dVector(np.array(unique_points)))


                # if translation_delta > 0.1 or rotation_delta > np.deg2rad(5):
                #     map_points = new_pcd.transform(self.odometry.copy()).points
                #     self.global_map.points.extend(map_points)

                trajectory.append(self.odometry[:3, 3].copy())
                self.XPOS, self.YPOS = self.odometry[0, 3], self.odometry[1, 3]
            except:
                return
        else: #for the very first frame
            self.local_map = get_map_2d(new_pcd, z_min=0.0, z_max=2.0)
            trajectory.append(self.odometry[:3, 3].copy())
            self.XPOS, self.YPOS = self.odometry[0, 3], self.odometry[1, 3]

    def perform_icp_point_to_plane(self, source, target):
        # to find surface normals -> needed for point to plane  
        # o3d.geometry.PointCloud.estimate_normals(
        #     source,
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)) #nearest neighbour search 
        # #the points are scattered and dont define a surface -> KDTree fits a plane through the points to visualize a surface

        # o3d.geometry.PointCloud.estimate_normals(
        #     target,
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

        threshold = 1.5 
        trans_init = self.transformation

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        return reg_p2p.transformation, reg_p2p.inlier_rmse

    def run(self):
        try:
            while True:
                file = './pcd/latest.pcd'

                new_pcd = o3d.io.read_point_cloud(file)
                if new_pcd.is_empty():
                    time.sleep(0.01)
                    continue
                new_pcd = new_pcd.voxel_down_sample(voxel_size=0.1)

                if not self.added:
                    self.pcd.points = new_pcd.points
                    # print(new_pcd.points)
                    self.vis.add_geometry(self.pcd)
                    self.added = True
                else:
                    self.pcd.points = new_pcd.points

                self.vis.update_geometry(self.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()

                # self.mapper.add_geometry(self.global_map)
                # print(f"Number of points in global map: {len(self.global_map.points)}")
                # self.mapper.update_geometry(self.global_map)
                # self.mapper.poll_events()
                # self.mapper.update_renderer()

                self.calc_traj(self.last_file, new_pcd, self.trajectory)
                # self.plot_thread.start()
                # print(f"Current Position: X={XPOS:.4f}, Y={YPOS:.4f}")

                self.last_file = file
                time.sleep(0.1)

            # time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting.")
            self.vis.destroy_window()
            # self.mapper.destroy_window()
            self.stop_event.set()
            self.pygame_thread.join()

if __name__ == "__main__":
    odom = LidarOdometry()
    odom.run()
