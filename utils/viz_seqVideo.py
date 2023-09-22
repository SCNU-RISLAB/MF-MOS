import open3d as o3d
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
class PointCloudPlayer():
    def __init__(self, velodyne_path, label_path, windows_w=1920, windows_h=1080,
                 background_color=[0, 0, 0], point_size=1.0, to_reset=True, color_cfg_path=None, viewfile_path="./viewfile.json"):
        self.velodyne_path = velodyne_path
        self.label_path = label_path

        self.windows_w = windows_w
        self.windows_h = windows_h
        self.background_color = background_color
        self.point_size = point_size
        self.to_reset = to_reset
        self.COLOR_CFG = yaml.safe_load(open(color_cfg_path, 'r'))
        self.viewfile_path = viewfile_path

        # self.get_vis()
        self.get_datas()
        self.set_colormap()

    def get_vis(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1920, height=1080)  # 创建窗口
        # self.vis.set_full_screen(True)
        # #设置连续帧 雷达第一视角

        # self.pcd = o3d.geometry.PointCloud()
        # self.vis.add_geometry(self.pcd)

        render_option = self.vis.get_render_option()  # 渲染配置
        render_option.background_color = np.array(self.background_color)  # 设置点云渲染参数，背景颜色
        render_option.point_size = self.point_size  # 设置渲染点的大小

        self.playing = True

        def exit_callback():
            print("exit")
            self.vis.close()
            quit()
            return True

        def pause_callback():
            print("key pause")
            self.playing = not self.playing
            return True

        self.vis.register_key_callback(ord("q"), exit_callback)
        self.vis.register_key_callback(ord("c"), pause_callback)

    def get_datas(self):
        self.pc_datas = []
        self.labels = []
        assert len(os.listdir(self.velodyne_path)) == len(os.listdir(self.label_path))

        for name in os.listdir(self.velodyne_path):
            pc_file = os.path.join(self.velodyne_path, name)
            label_file = os.path.join(self.label_path, name.replace(".bin", ".label"))
            self.pc_datas.append(np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4))
            self.labels.append(np.fromfile(label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF)

    def set_colormap(self):
        self.color_dict = self.COLOR_CFG["color_map"]
        moving_learning_map = self.COLOR_CFG["moving_learning_map"]
        movable_learning_map = self.COLOR_CFG["movable_learning_map"]

        for key in self.color_dict.keys():
            if (key == 250) or (key in movable_learning_map.keys() and movable_learning_map[key] == 2):
                self.color_dict[key] = [0, 0, 255]
                if key != 250 and moving_learning_map[key] == 2:
                    self.color_dict[key] = [255, 0, 0]
            else:
                self.color_dict[key] = [255, 255, 255]

    def draw_color(self, point_xyz, label):
        colors = np.zeros((point_xyz.shape[0], 3))
        for i in list(set(label.tolist())):
            if i not in self.color_dict.keys():
                colors[label == i] = np.array([0, 255, 0])
                continue
            colors[label == i] = np.array(self.color_dict[i])

        return colors

    def save_view_file(self, pcd_numpy):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
        vis.add_geometry(pcd)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        vis.add_geometry(axis)
        vis.run()  # user changes the view and press "q" to terminate
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(self.viewfile_path, param)
        vis.destroy_window()

    def play(self):
        if not os.path.exists(self.viewfile_path):
            self.save_view_file(- self.pc_datas[0][:, :3])

        pcds = []
        self.get_vis()
        with tqdm(total=len(self.pc_datas)) as pbar:
            pbar.set_description("Reading point cloud...")
            for pc_data, label in zip(self.pc_datas, self.labels):
                point_xyz = - pc_data[:, :3]  # x, y, z

                pcd = o3d.open3d.geometry.PointCloud()  # 创建点云对象
                pcd.points = o3d.utility.Vector3dVector(point_xyz)
                pcd.colors = o3d.utility.Vector3dVector(self.draw_color(point_xyz, label))

                pcds.append(pcd)
                # self.vis.add_geometry(pcd)

                pbar.update(1)

        ctr = self.vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(self.viewfile_path)
        ctr.convert_from_pinhole_camera_parameters(param)

        for pcd in pcds:
            self.vis.clear_geometries()
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            # self.vis.add_geometry(axis)
            self.vis.add_geometry(pcd)
            # self.vis.poll_events()
            # self.vis.update_renderer()
            ctr.convert_from_pinhole_camera_parameters(param)
            time.sleep(0.2)
            self.vis.run()
        self.vis.destroy_window()


if __name__ == "__main__":
    use_pred = True
    dataset_root_path = "/data1/MotionSeg3d_Dataset/data_odometry_velodyne/dataset/sequences"
    label_root_path = "/data1/MF-MOS/log/Valid/predictions/sequences"
    # label_root_path = "/data1/MotionSeg3d_Dataset/data_odometry_velodyne/dataset/sequences"

    seq = "07"
    if use_pred:
        labels_path = os.path.join(label_root_path, seq, "predictions_fuse")
    else:
        labels_path = os.path.join(label_root_path, seq, "labels")
    pcp = PointCloudPlayer(velodyne_path=os.path.join(dataset_root_path, seq, "velodyne"),
                           label_path=labels_path,
                           color_cfg_path="/data1/MF-MOS/config/labels/semantic-kitti-mos.raw.yaml",
                           point_size=1.2)

    pcp.play()