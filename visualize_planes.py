import open3d as o3d 
import numpy as np
import random
import json

class PlaneRenderer():

    def __init__(self, planes, bbox=[-100,-100,-100,100,100,100]):
        # plane are defined by a normal vector and a offset along the normal direction
        # eg. [1,0,0,10] defines a plane parallel to yz plane with offset 10
        # planes N*4
        self.planes = np.asarray(planes)
        self.bbox = bbox
        self.resolution = 0.5
        self.mesh = []

    def sample_points(self):
        vis_plane = []
        for plane in self.planes:
            # If the plane are oriented vertically, filp it to calculate and filp back
            normal = plane[:3]
            index = np.abs(normal).argsort()
            reverse_index = np.argsort(index)
            plane[:3] = normal[index]
            points = []

            for x in [self.bbox[0], self.bbox[3]]:
                for y in [self.bbox[1], self.bbox[4]]:
                    z = self.calculate(plane, x, y)
                    points.append(np.asarray([x,y,z])[reverse_index])

            color = tuple(np.random.randint(256, size=3))
            color = [np.asarray(color)/255.0]*4
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(points))
            mesh.vertex_colors = o3d.utility.Vector3dVector(color)
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0,1,2],[2,1,0],[1,3,2],[1,2,3]]))
            vis_plane.append(mesh)
        self.mesh = vis_plane
        return self.mesh

    def visualize(self):
        if len(self.mesh)==0:
            self.sample_points()
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
        o3d.visualization.draw_geometries([axis]+self.mesh)
        return

    def calculate(self, plane, x, y):
        return (plane[3]-plane[0]*x-plane[1]*y)/plane[2]

    def export_ply(self,export_path):
        if len(self.mesh)==0:
            self.sample_points()

        vertices = []
        faces = []
        colors = []
        index_offset = 0

        for mesh in self.mesh:
            vertices.append(np.asarray(mesh.vertices))
            colors.append(np.asarray(mesh.vertex_colors))
            faces.append(np.asarray(mesh.triangles)+index_offset)
            index_offset += np.asarray(mesh.vertices).shape[0]

        faces = np.concatenate(faces)
        vertices = np.concatenate(vertices)
        colors = np.concatenate(colors)

        with open(export_path,'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % vertices.shape[0])
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("element face %d\n" % faces.shape[0])
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for i in range(vertices.shape[0]):
                f.write(" ".join(list(map(str,vertices[i].tolist())))+" ")
                f.write(" ".join(list(map(str,(colors[i]*255).astype(int).tolist())))+"\n")
            for i in range(faces.shape[0]):
                f.write("3 " + " ".join(list(map(str,faces[i].tolist())))+"\n")


def parse_planes(file_path):
    with open(file_path,'r') as f:
        data = json.load(f)
    planes_data = data['planes']
    planes = []
    for p in planes_data:
        offset = p['offset']
        planes.append(np.asarray(p['normal']+[offset/100.0]))
    return planes


if __name__ == "__main__":
    # scene = '/media/daxuan/DATA/Dataset/Structured3d/Structured3D/scene_00000'
    planes = parse_planes('/media/daxuan/DATA/Dataset/Structured3d/Structured3D/scene_00001/annotation_3d.json')
    renderer = PlaneRenderer(planes)
    renderer.visualize()
    renderer.export_ply('planes.ply')
