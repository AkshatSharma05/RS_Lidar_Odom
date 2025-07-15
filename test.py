import open3d as o3d
import time
import os
import glob

pcd_dir = "./pcd"
seen = set()
last_file = None

vis = o3d.visualization.Visualizer()

vis.create_window("LIDAR")
pcd = o3d.geometry.PointCloud()
vis.get_render_option().background_color = [0, 0, 0]  
added = False

print("Watching directory:", pcd_dir)

try:
    while True:
        # Get sorted list of .pcd files
        pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

        for file in pcd_files:
            if file not in seen:
                print("New file:", file)
                seen.add(file)

                new_pcd = o3d.io.read_point_cloud(file)
                # new_pcd = new_pcd.voxel_down_sample(voxel_size=0.01)

                if not added:
                    pcd.points = new_pcd.points
                    print(new_pcd.points)
                    vis.add_geometry(pcd)
                    added = True
                else:
                    pcd.points = new_pcd.points
                    vis.update_geometry(pcd)

                vis.poll_events()
                vis.update_renderer()

                if last_file and os.path.exists(last_file):
                    try:
                        os.remove(last_file)
                        # print(f"Deleted: {last_file}")
                    except Exception as e:
                        print(f"Error deleting {last_file}: {e}")

                last_file = file
                time.sleep(0.02)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting.")
    vis.destroy_window()
