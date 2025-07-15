import open3d as o3d
import time
import os
# import glob
import numpy as np
import threading
import pygame

class LidarOdometry:
    def __init__(self, pcd_dir="./pcd"):
        self.pcd_dir = pcd_dir
        self.seen = set()
        self.last_file = None
        self.prev_file = None  

        self.XPOS = 0.0
        self.YPOS = 0.0
        self.frame_skip = 0
        self.frame_count = 0

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("LIDAR")
        self.pcd = o3d.geometry.PointCloud()
        self.vis.get_render_option().background_color = [0, 0, 0]  
        self.added = False

        self.transformation = np.eye(4) #identity matrix of size 4x4
        self.odometry = np.eye(4)
        self.trajectory = [] 

        self.stop_event = threading.Event()
        self.pygame_thread = threading.Thread(target=self.pygame_plotter, args=(self.get_pos, self.stop_event))
        self.pygame_thread.start()

    def get_pos(self):
        return self.XPOS, self.YPOS

    def pygame_plotter(self, get_pos, stop_event):
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

    def remove(self, prev_file):
        if prev_file and prev_file != self.last_file and os.path.exists(prev_file):
            try:
                os.remove(prev_file)
                # print(f"Deleted: {prev_file}")
            except Exception as e:
                print(f"Error deleting {prev_file}: {e}")

    def calc_traj(self, last_file, new_pcd, trajectory):
        if last_file is not None:
            last_pcd = o3d.io.read_point_cloud(last_file)
            last_pcd = last_pcd.voxel_down_sample(voxel_size=0.2)
            try:
                transformation, inlier_rmse3 = self.perform_icp_point_to_plane(last_pcd, new_pcd)
                if not np.allclose(transformation, np.eye(4), atol=1e-2): #allclose checks if two matrices are equal
                    proposed = np.dot(self.odometry, transformation) #next pose estimate

                    #odometry[:2,3] -> first two elements of last column

                    if np.linalg.norm(proposed[:2, 3] - self.odometry[:2, 3]) > 0.5: #distance between x and y coords of the poses
                        return 
                    self.odometry = np.dot(self.odometry, transformation) # = proposed
                trajectory.append(self.odometry[:3, 3].copy())
                self.XPOS, self.YPOS = self.odometry[0, 3], self.odometry[1, 3]
            except:
                return
        else: #for the very first frame
            trajectory.append(self.odometry[:3, 3].copy())
            self.XPOS, self.YPOS = self.odometry[0, 3], self.odometry[1, 3]

    def perform_icp_point_to_plane(self, source, target):
        # to find surface normals -> needed for point to plane  
        o3d.geometry.PointCloud.estimate_normals(
            source,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)) #nearest neighbour search 
        #the points are scattered and dont define a surface -> KDTree fits a plane through the points to visualiza surface

        o3d.geometry.PointCloud.estimate_normals(
            target,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

        threshold = 1.5 
        trans_init = self.transformation

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return reg_p2p.transformation, reg_p2p.inlier_rmse

    def run(self):
        try:
            while True:
                file = './pcd/latest.pcd'

                if not os.path.exists(file) or os.path.getsize(file) < 100:
                    # time.sleep(0.01)
                    continue
                # pcd_files = sorted(glob.glob(os.path.join(self.pcd_dir, "*.pcd")))
                # for file in pcd_files:
                #     if file not in self.seen:
                    # print("New file:", file)
                self.seen.add(file)

                new_pcd = o3d.io.read_point_cloud(file)
                if new_pcd.is_empty():
                    # time.sleep(0.01)
                    continue
                new_pcd = new_pcd.voxel_down_sample(voxel_size=0.2)

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

                self.calc_traj(self.last_file, new_pcd, self.trajectory)
                # threading.Thread(target=self.remove, args=(self.prev_file,), daemon=True).start()
                # print(f"Current Position: X={XPOS:.4f}, Y={YPOS:.4f}")

                self.prev_file = self.last_file
                self.last_file = file
                # time.sleep(0.1)

            # time.sleep(0.001)
        except KeyboardInterrupt:
            print("Exiting.")
            self.vis.destroy_window()
            self.stop_event.set()

if __name__ == "__main__":
    odom = LidarOdometry()
    odom.run()
