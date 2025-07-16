import open3d as o3d
import time
import numpy as np
import threading
import pygame
from collections import defaultdict

class LidarOdometry:
    def __init__(self, pcd_dir="./pcd"):
        self.pcd_dir = pcd_dir
        self.last_file = None

        self.XPOS = 0.0
        self.YPOS = 0.0
        self.frame_skip = 0
        self.frame_count = 0

        self.local_map = None
        self.global_map_2d = None

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("LIDAR")
        self.pcd = o3d.geometry.PointCloud()
        self.vis.get_render_option().background_color = [0, 0, 0]  
        self.added = False

        self.global_map = o3d.geometry.PointCloud()
        self.voxel_map = {}
        self.voxel_counter = defaultdict(int)
        self.mapper = o3d.visualization.Visualizer()
        self.mapper.create_window("Map")
        
        self.mapper.get_render_option().background_color = [0, 0, 0]
        
        self.current_frame = 0
        self.decay_threshold = 15
        self.min_hits = 100

        self.transformation = np.eye(4) #identity matrix of size 4x4
        self.odometry = np.eye(4)
        self.trajectory = [] 

        self.traj_line = o3d.geometry.LineSet()
        self.traj_points = []
        self.traj_indices = []
        self.traj_added = False


        self.stop_event = threading.Event()
        self.pygame_thread = threading.Thread(target=self.pygame_plotter, args=(self.get_pos,  self.get_global_map, self.stop_event))
        self.pygame_thread.start()

    def get_pos(self):
        return self.XPOS, self.YPOS

    def get_global_map(self):
        return self.global_map_2d

    def pygame_plotter(self, get_pos, get_map, stop_event):
        pygame.init()
        show_odom = True
        size = 600
        screen = pygame.display.set_mode((size, size)) 
        pygame.display.set_caption("Odometry + 2D Lidar Map")
        clock = pygame.time.Clock()

        white = (255, 255, 255)
        blue = (0, 0, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)

        points = []

        world_min = -10
        world_max = 10

        def world_to_screen(x, y):
            sx = int((x - world_min) / (world_max - world_min) * size)
            sy = int(size - (y - world_min) / (world_max - world_min) * size)
            return sx, sy

        while not stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()
                    return

            screen.fill(white)
            
            if show_odom:
            # === Draw odometry ===
                xpos, ypos = get_pos()
                points.append(world_to_screen(xpos, ypos))
                if len(points) > 1:
                    pygame.draw.lines(screen, blue, False, points, 2)
                for pt in points:
                    pygame.draw.circle(screen, black, pt, 2)

                # === Draw lidar map (in global frame) ===
                global_map = get_map()
                if global_map is not None:
                    for pt in global_map:
                        x, y = pt[0], pt[1]
                        sx, sy = world_to_screen(x, y)
                        if 0 <= sx < size and 0 <= sy < size:
                            pygame.draw.circle(screen, red, (sx, sy), 1)

            if self.current_frame % 50 == 0: 
                show_odom = False
                screen.fill(white)
                global_map = get_map()
                if global_map is not None:
                    for pt in global_map:
                        x, y = pt[0], pt[1]
                        sx, sy = world_to_screen(x, y)
                        if 0 <= sx < size and 0 <= sy < size:
                            pygame.draw.circle(screen, red, (sx, sy), 1)

                pygame.display.flip()
                pygame.image.save(screen, f"map.png")
                show_odom = True

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


    def calc_traj(self, last_file, new_pcd, trajectory):
        def get_map_2d(pcd, z_min=0.0, z_max=2.0):
            arr = np.asarray(pcd.points)
            mask = (arr[:, 2] >= z_min) & (arr[:, 2] <= z_max)
            arr_2d = arr[mask][:, :2]
            return arr_2d
        
        def update_voxel_map(voxel_map, global_map, new_pcd, pose, current_frame,
                         voxel_size=0.1, min_hits=50, decay_thresh=20):
            transformed_points = np.asarray(new_pcd.transform(pose).points)

            for pt in transformed_points:
                key = tuple((pt / voxel_size).astype(int))
                if key in voxel_map:
                    voxel_map[key][0] += 1
                    voxel_map[key][1] = current_frame
                    if voxel_map[key][0] >= min_hits:
                        voxel_map[key][2] = True
                else:
                    voxel_map[key] = [1, current_frame, False]  # seen_count, last_seen, is_static

            keys_to_remove = []
            for key, (count, last_seen, is_static) in voxel_map.items():
                if not is_static and current_frame - last_seen > decay_thresh:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del voxel_map[key]

            static_points = [np.array(key, dtype=float) * voxel_size
                            for key, val in voxel_map.items() if val[2]]
            global_map.points = o3d.utility.Vector3dVector(np.array(static_points))
        
        if last_file is not None:
            last_pcd = o3d.io.read_point_cloud(last_file)
            last_pcd = last_pcd.voxel_down_sample(voxel_size=0.1)
            try:
                self.global_map_2d = get_map_2d(self.global_map, z_min=-0.8, z_max=-0.2)

                transformation, _ = self.perform_icp_point_to_plane(last_pcd, new_pcd)
                if not np.allclose(transformation, np.eye(4), atol=1e-2): #allclose checks if two matrices are equal
                    self.odometry = np.dot(self.odometry, transformation) # = proposed

                self.current_frame += 1
                update_voxel_map(self.voxel_map, self.global_map, new_pcd,
                             self.odometry.copy(), self.current_frame,
                             voxel_size=0.2, min_hits=self.min_hits,
                             decay_thresh=self.decay_threshold)

                trajectory.append(self.odometry[:3, 3].copy())
                self.XPOS, self.YPOS = self.odometry[0, 3], self.odometry[1, 3]


                # self.traj_points.append(self.odometry[:3, 3].copy())
                # traj_xyz = o3d.utility.Vector3dVector(self.traj_points)
                # self.traj_line.points = traj_xyz

                # # Update line indices if we have at least 2 points
                # if len(self.traj_points) > 1:
                #     self.traj_indices = [[i - 1, i] for i in range(1, len(self.traj_points))]
                #     self.traj_line.lines = o3d.utility.Vector2iVector(self.traj_indices)

                # if not self.traj_added:
                #     self.vis.add_geometry(self.traj_line)
                #     self.traj_added = True
                # else:
                #     self.vis.update_geometry(self.traj_line)
            except:
                return
        else: #for the very first frame
            trajectory.append(self.odometry[:3, 3].copy())
            self.traj_points.append(self.odometry[:3, 3].copy())
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

        threshold = 10.0
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

                self.mapper.add_geometry(self.global_map)
                self.vis.update_geometry(self.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()

                print(f"Number of points in global map: {len(self.global_map.points)}")
                self.mapper.update_geometry(self.global_map)
                self.mapper.poll_events()
                self.mapper.update_renderer()

                self.calc_traj(self.last_file, new_pcd, self.trajectory)
                # self.plot_thread.start()
                # print(f"Current Position: X={XPOS:.4f}, Y={YPOS:.4f}")

                self.last_file = file
                time.sleep(0.1)

            # time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting.")
            self.vis.destroy_window()
            self.mapper.destroy_window()
            self.stop_event.set()

if __name__ == "__main__":
    odom = LidarOdometry()
    odom.run()
