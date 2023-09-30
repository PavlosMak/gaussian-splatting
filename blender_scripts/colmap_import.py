import struct
import numpy as np
import bpy
import collections

from mathutils import Vector

Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

PATH_TO_MODEL_FILE = "../colmap_test/sparse/0/points3D.bin"
OUTPUT = "../colmap_test/sparse/0/points3D_test.bin"

# methods adapted from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    print(f"Reading from {path_to_model_file}")
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def points3D_to_mesh(points3D, name="Point Cloud"):
    # create the object
    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name, mesh)

    # add custom attributes to object
    obj.data.attributes.new(name="luminance", type='FLOAT', domain='POINT')
    obj.data.attributes.new(name="rgb", type='FLOAT_VECTOR', domain='POINT')
    obj.data.attributes.new(name="error", type='FLOAT', domain='POINT')
    obj.data.attributes.new(name="image_ids", type='STRING', domain='POINT')
    obj.data.attributes.new(name="point2D_idxs", type='STRING', domain='POINT')

    # create the lists in which we will record data
    xyzs = []
    rgbs = []
    errs = []
    image_ids = []
    point2D_idxs = []
    energies = []

    # loop over the points
    for _, pt in points3D.items():
        xyzs.append(pt.xyz.tolist())
        rgbs.append(pt.rgb)

        errs.append(pt.error)
        image_ids.append(pt.image_ids.tolist())
        point2D_idxs.append(pt.point2D_idxs.tolist())
        energies.append(0.299 * pt.rgb[0] + 0.587 * pt.rgb[1] + 0.114 * pt.rgb[2])

    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # make the mesh
    mesh.from_pydata(xyzs, [], [])

    # set the attributes
    for i, _ in enumerate(xyzs):
        obj.data.attributes['luminance'].data[i].value = energies[i]
        obj.data.attributes['error'].data[i].value = errs[i]
        obj.data.attributes['rgb'].data[i].vector = Vector(rgbs[i])
        # since Blender does not let us have arbitrary lenght sequences
        # we do this hacky thing for the next two fields. It gets worse
        # because the strings get truncated so we are losing data here.
        obj.data.attributes['image_ids'].data[i].value = str(image_ids[i])
        obj.data.attributes['point2D_idxs'].data[i].value = str(point2D_idxs[i])

# Load the model
points3d = read_points3D_binary(PATH_TO_MODEL_FILE)
points3D_to_mesh(points3d, name="Point Cloud")

print("Loaded COLMAP bin!")
