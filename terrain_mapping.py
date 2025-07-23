import open3d as o3d
import time
import numpy as np
import cv2
import json
from collections import defaultdict
import os

class LidarOdometry:
    def __init__(self, pcd_dir="./pcd", save_interval=10):
        self.pcd_dir = pcd_dir
        self.last_file = None

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("LIDAR")
        self.pcd = o3d.geometry.PointCloud()
        self.vis.get_render_option().background_color = [0, 0, 0]
        self.added = False

        self.global_map = o3d.geometry.PointCloud()
        self.voxel_map = {}
        self.voxel_counter = defaultdict(int)
        self.voxel_size = 0.1
        # self.mapper = o3d.visualization.Visualizer()
        # self.mapper.create_window("Map")
        # self.mapper.get_render_option().background_color = [0, 0, 0]

        self.current_frame = 0
        self.decay_threshold = 15
        self.min_hits = 150

        self.transformation = np.eye(4)
        self.odometry = np.eye(4)
        self.trajectory = []
        self.traj_points = []
        self.traj_indices = []

        self.map_resolution = 0.1  # meters per pixel
        self.map_size = 100        # meters (width and height in world coordinates)
        self.global_map_size = int(self.map_size / self.map_resolution)  # image dimensions in pixels
        # Initialize the global elevation map with NaNs.
        self.global_elevation_map = np.full((self.global_map_size, self.global_map_size), np.nan)
        self.origin_offset = self.global_map_size // 2  # pixel coordinate corresponding to world (0,0)
        
        self.save_interval = save_interval
        self.last_save_time = time.time()

    def get_pos(self):
        return self.XPOS, self.YPOS

    def get_global_map(self):
        return self.global_map

    def calc_traj(self, last_file, new_pcd, trajectory):
        def update_voxel_map(voxel_map, global_map, new_pcd, pose, current_frame,
                            voxel_size=0.2, min_hits=100, decay_thresh=15, neighbor_margin=2):
            """
            Update voxel-based global map with new point cloud data.

            Args:
                voxel_map (dict): Key = voxel coord tuple; Value = [hit_count, last_seen_frame, is_static]
                global_map (o3d.geometry.PointCloud): The Open3D global point cloud
                new_pcd (o3d.geometry.PointCloud): New point cloud from LiDAR/frame
                pose (np.ndarray): 4x4 transformation matrix
                current_frame (int): Current frame index
                voxel_size (float): Size of a voxel in meters
                min_hits (int): Minimum hits before marking as static
                decay_thresh (int): Frames before a non-static voxel is removed
                neighbor_margin (int): Distance in voxel units to skip updates around static voxels
            """
            transformed_points = np.asarray(new_pcd.transform(pose).points)

            # Convert static voxel keys to a set for fast neighborhood checking
            static_keys = {key for key, val in voxel_map.items() if val[2]}

            for pt in transformed_points:
                key = tuple((pt / voxel_size).astype(int))

                should_skip = False
                for dx in range(-neighbor_margin, neighbor_margin + 1):
                    for dy in range(-neighbor_margin, neighbor_margin + 1):
                        for dz in range(-neighbor_margin, neighbor_margin + 1):
                            neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                            if neighbor_key in static_keys:
                                should_skip = True
                                break
                        if should_skip:
                            break
                    if should_skip:
                        break
                if should_skip:
                    continue

                if key in voxel_map:
                    hit_count, last_seen, is_static = voxel_map[key]
                    if not is_static:
                        hit_count += 1
                        last_seen = current_frame
                        if hit_count >= min_hits:
                            is_static = True
                        voxel_map[key] = [hit_count, last_seen, is_static]
                else:
                    voxel_map[key] = [1, current_frame, False]

            # Remove decayed dynamic voxels
            keys_to_remove = [key for key, (count, last_seen, is_static) in voxel_map.items()
                            if not is_static and (current_frame - last_seen > decay_thresh)]
            for key in keys_to_remove:
                del voxel_map[key]

            # Update global map with only static points
            static_points = [np.array(key, dtype=float) * voxel_size
                            for key, val in voxel_map.items() if val[2]]
            if static_points:
                global_map.points = o3d.utility.Vector3dVector(np.array(static_points))



        if last_file is not None:
            last_pcd = o3d.io.read_point_cloud(last_file)
            last_pcd = last_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            try:
                transformation, _ = self.perform_icp_point_to_plane(last_pcd, new_pcd)
                if not np.allclose(transformation, np.eye(4), atol=1e-2):
                    self.odometry = np.dot(self.odometry, transformation)
                    print(self.odometry)

                self.current_frame += 1
                update_voxel_map(self.voxel_map, self.global_map, new_pcd,
                                 self.odometry.copy(), self.current_frame,
                                 voxel_size=self.voxel_size, min_hits=self.min_hits,
                                 decay_thresh=self.decay_threshold)

                trajectory.append(self.odometry[:3, 3].copy())
                self.XPOS, self.YPOS = self.odometry[0, 3], self.odometry[1, 3]

            except Exception as e:
                print("ICP Exception:", e)
                return
        else:
            trajectory.append(self.odometry[:3, 3].copy())
            self.traj_points.append(self.odometry[:3, 3].copy())
            self.XPOS, self.YPOS = self.odometry[0, 3], self.odometry[1, 3]

    def perform_icp_point_to_plane(self, source, target):
        threshold = 10.0
        trans_init = self.transformation

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p.transformation, reg_p2p.inlier_rmse

    def update_global_elevation_map_from_static_voxels(self):
        self.global_elevation_map[:] = np.nan
        for key, (count, last_seen, is_static) in self.voxel_map.items():
            if is_static:
                x, y, z = np.array(key, dtype=float) * self.voxel_size
                gx = int(x / self.map_resolution) + self.origin_offset
                gy = int(y / self.map_resolution) + self.origin_offset
                if 0 <= gx < self.global_map_size and 0 <= gy < self.global_map_size:
                    current_val = self.global_elevation_map[gy, gx]
                    if np.isnan(current_val) or z < current_val:
                        self.global_elevation_map[gy, gx] = z

    def show_global_elevation_map(self):
        map_copy = self.global_elevation_map.copy()
        if np.all(np.isnan(map_copy)):
            return

        min_val = np.nanmin(map_copy)
        map_copy = np.nan_to_num(map_copy, nan=min_val)

        norm_map = (map_copy - np.min(map_copy)) / (np.max(map_copy) - np.min(map_copy) + 1e-5)
        norm_map = (norm_map * 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(norm_map, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        smoothed = cv2.GaussianBlur(closed, (5, 5), 0)

        color_map = cv2.applyColorMap(smoothed, cv2.COLORMAP_JET)
        cv2.imshow("Global Elevation Map", color_map)
        cv2.waitKey(1)

    def save_global_map(self):
        map_copy = self.global_elevation_map.copy()
        if np.all(np.isnan(map_copy)):
            print("No valid map data to save.")
            return

        min_val = np.nanmin(map_copy)
        map_copy = np.nan_to_num(map_copy, nan=min_val)
        norm_map = (map_copy - np.min(map_copy)) / (np.max(map_copy) - np.min(map_copy) + 1e-5)
        occ_img = (norm_map * 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        occ_img = cv2.dilate(occ_img, kernel, iterations=1)
        occ_img = cv2.morphologyEx(occ_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        occ_img = cv2.GaussianBlur(occ_img, (5, 5), 0)

        save_dir = "./saved_maps"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time())

        img_filename = os.path.join(save_dir, f"occ_map.png")
        cv2.imwrite(img_filename, occ_img)

        metadata = {
            "map_resolution_m_per_pixel": self.map_resolution,
            "global_map_size_pixels": self.global_map_size,
            "origin_pixel": [self.origin_offset, self.origin_offset],
            "map_size_m": self.map_size
        }
        metadata_filename = os.path.join(save_dir, f"occ_map_metadata.json")
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Saved occupancy map to {img_filename} and metadata to {metadata_filename}")

    def run(self):
        try:
            while True:
                file = './pcd/latest.pcd'

                new_pcd = o3d.io.read_point_cloud(file)
                if new_pcd.is_empty():
                    time.sleep(0.01)
                    continue
                new_pcd = new_pcd.voxel_down_sample(voxel_size=self.voxel_size)

                if not self.added:
                    self.pcd.points = new_pcd.points
                    self.vis.add_geometry(self.pcd)
                    self.added = True
                else:
                    self.pcd.points = new_pcd.points

                self.vis.update_geometry(self.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()

                # self.mapper.add_geometry(self.global_map)
                # self.mapper.update_geometry(self.global_map)
                # self.mapper.poll_events()
                # self.mapper.update_renderer()

                # print(f"Number of points in global map: {len(self.global_map.points)}")

                self.calc_traj(self.last_file, new_pcd, self.trajectory)
                self.update_global_elevation_map_from_static_voxels()
                self.show_global_elevation_map()

                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    self.save_global_map()
                    self.last_save_time = current_time

                self.last_file = file
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Exiting.")
            self.vis.destroy_window()
            self.mapper.destroy_window()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    odom = LidarOdometry()
    odom.run()
