import argparse
import cv2
import math
import numpy as np
import open3d
import os
from sklearn.preprocessing import normalize
from tqdm import tqdm


class PointCloudReader():

    def __init__(self, path, resolution="full", random_level=0, generate_color=False, generate_normal=False):
        self.path = path
        self.random_level = random_level
        self.resolution = resolution
        self.generate_color = generate_color
        self.generate_normal = generate_normal
        sections = [p for p in os.listdir(os.path.join(path, "2D_rendering"))]
        self.depth_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", self.resolution, "depth.png"]) for p in sections]
        self.rgb_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", self.resolution, "rgb_coldlight.png"]) for p in sections]
        self.normal_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", self.resolution, "normal.png"]) for p in sections]
        self.camera_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", "camera_xyz.txt"]) for p in sections]
        self.camera_centers = self.read_camera_center()
        self.point_cloud = self.generate_point_cloud(self.random_level, color=self.generate_color, normal=self.generate_normal)

    def read_camera_center(self):
        camera_centers = []
        for i in range(len(self.camera_paths)):
            with open(self.camera_paths[i], 'r') as f:
                line = f.readline()
            center = list(map(float, line.strip().split(" ")))
            camera_centers.append(np.asarray([center[0], center[1], center[2]]))
        return camera_centers

    def filp_if_needed(self, point_coord, camera_center, normal):
        offset = point_coord - camera_center
        offset[1], offset[2] = offset[2],offset[1]
        if offset.dot(normal) > 0:
            return -normal
        else:
            return normal

    def filp(self, point_coord, camera_center, normal):
        offset = point_coord - camera_center
        offset[1], offset[2] = offset[2],offset[1]
        if offset.dot(normal) > 0:
            return True
        else:
            return False


    def generate_point_cloud(self, random_level=0, color=False, normal=False):
        coords = []
        colors = []
        normals = []
        points = {}
        # Getting Coordinates
        for i in range(len(self.depth_paths)):
            # print(self.depth_paths[i])
            depth_img = cv2.imread(self.depth_paths[i], cv2.IMREAD_ANYDEPTH)
            x_tick = 180.0/depth_img.shape[0]
            y_tick = 360.0/depth_img.shape[1]
            for x in range(depth_img.shape[0]):
                for y in range(depth_img.shape[1]):
                    # x * x_tick 0-180
                    # need 90 - -09
                    alpha = 90 - (x * x_tick)
                    beta = y * y_tick -180
                    depth = depth_img[x,y] + np.random.random()*random_level
                    z_offset = depth*np.sin(np.deg2rad(alpha))
                    xy_offset = depth*np.cos(np.deg2rad(alpha))
                    x_offset = xy_offset * np.sin(np.deg2rad(beta))
                    y_offset = xy_offset * np.cos(np.deg2rad(beta))
                    point = np.asarray([x_offset, y_offset, z_offset])
                    coords.append(point + self.camera_centers[i])

            if color:
                rgb_img = cv2.imread(self.rgb_paths[i])
                for x in range(rgb_img.shape[0]):
                    for y in range(rgb_img.shape[1]):
                        colors.append(rgb_img[x, y])

            if normal:
                normal_img = cv2.imread(self.normal_paths[i])
                for x in range(normal_img.shape[0]):
                    for y in range(normal_img.shape[1]):
                        normals.append((normal_img[x, y]-127.0)/127.0)
                for j in range(len(normals)):
                    # Normal map may not be correct, filp if we need to
                    normals[j] = self.filp_if_needed(coords[j], self.camera_centers[i], normals[j])

        points['colors'] = np.asarray(colors)/255.0
        points['coords'] = np.asarray(coords)
        points['normals'] = np.asarray(normals)
        return points

    def get_point_cloud(self):
        return self.point_cloud

    def visualize(self):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(self.point_cloud['coords'])
        if self.generate_normal:
            pcd.normals = open3d.utility.Vector3dVector(self.point_cloud['normals'])
        if self.generate_color:
            pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['colors'])
        open3d.visualization.draw_geometries([pcd])

    def get_open3d_ppointcloud(self):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(self.point_cloud['coords'])
        if self.generate_normal:
            pcd.normals = open3d.utility.Vector3dVector(self.point_cloud['normals'])
        if self.generate_color:
            pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['colors'])
        return pcd

    def export_ply(self, path):
        # Save ASCII ply file for visulization
        '''
        ply
        format ascii 1.0
        element vertex 259200
        property float x
        property float y
        property float z
        property uchar r
        property uchar g
        property uchar b
        property float nx
        property float ny
        property float nz
        end_header
        '''
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % self.point_cloud['coords'].shape[0])
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if self.generate_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            if self.generate_normal:
                f.write("property float nx\n")
                f.write("property float nz\n")
                f.write("property float ny\n")
            f.write("end_header\n")
            for i in range(self.point_cloud['coords'].shape[0]):
                normal = []
                color = []
                coord = self.point_cloud['coords'][i].tolist()
                if self.generate_color:
                    color = list(map(int, (self.point_cloud['colors'][i]*255).tolist()))
                if self.generate_normal:
                    normal = self.point_cloud['normals'][i].tolist()
                data = coord + color + normal
                f.write(" ".join(list(map(str,data)))+'\n')

    def export_npy(self, path):
        np.save(path, self.point_cloud)
        return path

def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 2D Layout Visualization")
    parser.add_argument("--data_path", required=True,
                        help="dataset path", metavar="DIR")
    parser.add_argument("--save_path", required=True,
                        help="save pointclouds to path", metavar="DIR")
    parser.add_argument("--scene", required=False, default=-1,
                        help="scene id, -1 for all scenes", type=int)
    parser.add_argument("--room", required=False, default=-1,
                        help="room id, -1 for all rooms registerd together", type=int)
    parser.add_argument('--export_ply', dest='ply', action='store_true')
    parser.add_argument('--export_npy', dest='npy', action='store_true')

    return parser.parse_args()


def print_args(args):
    print("Dataset Path: %s" % args.data_path)
    print("Saving Path: %s" % args.data_path)
    if args.scene == -1:
        print("All scene will be generated...")
    else:
        print("Scene %d will be generated...")
    if not args.ply:
        print("No ply file will be generate")
    else:
        print("PLY file will be generate")
    if not args.npy:
        print("No numpy file will be generate")
    else:
        print("Numpy file will be generate")

def prepare_dir(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

def main():
    args = parse_args()
    scene_path = args.data_path
    prepare_dir(args)
    processed_scenes = [i.split('.')[0] for i in os.listdir(args.save_path)]
    scenes = [os.path.join(scene_path, i) for i in os.listdir(scene_path)]

    for scene in tqdm(scenes):
        if scene in processed_scenes:
            continue

        reader = PointCloudReader(scene, random_level=10, generate_color=True, generate_normal=False)

        if args.npy:
            filename = os.path.basename(scene)+".npy"
            reader.export_npy(os.path.join(args.save_path, filename))
        if args.ply:
            filename = os.path.basename(scene)+".ply"
            reader.export_ply(os.path.join(args.save_path, filename))

if __name__ == "__main__":
    main()
