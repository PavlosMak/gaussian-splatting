
import struct
import bpy
import collections

Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

PATH_TO_MODEL_FILE = "../colmap_test/sparse/0/points3D.bin"
OUTPUT = "../colmap_test/sparse/0/points3D_test.bin"

#methods adapted from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")

def select_object_by_name(object_name):
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    # Select the object by name
    bpy.data.objects[object_name].select_set(True)
    # Set the active object
    bpy.context.view_layer.objects.active = bpy.data.objects[object_name]


def write_pointcloudmesh_binary(object_name: str, output_path: str):
    select_object_by_name(object_name)
    obj = bpy.context.object
    if obj and obj.type == 'MESH':
        print(f"Saving {object_name}")
        # Iterate over the vertices of the selected mesh
        with open(output_path, "wb") as fid:
            attributes = obj.data.attributes
            vertices = obj.data.vertices
            write_next_bytes(fid, len(vertices), "Q")
            for i, vertex in enumerate(vertices):
                i = vertex.index
                write_next_bytes(fid, i, "Q")
                xyz = list(vertex.co)
                write_next_bytes(fid, xyz, "ddd")
                rgb = [int(channel) for channel in attributes['rgb'].data[i].vector]
                write_next_bytes(fid, rgb, "BBB")
                err = attributes['error'].data[i].value
                write_next_bytes(fid, err, "d")
                image_ids = [0,0,0]
                track_length = len(image_ids)
                write_next_bytes(fid, track_length, "Q")
                point2D_idxs = [0,0,0]
                for image_id, point2D_id in zip(image_ids, point2D_idxs):
                    write_next_bytes(fid, [image_id, point2D_id], "ii")
    else:
        print("Selected object is not a mesh.")
    print(f"Saved {object_name}")

print("WARNING: This method does not persist the image ids and the 2D point indexes associated with each point.")

OBJECT_NAME = "Point Cloud"
write_pointcloudmesh_binary(OBJECT_NAME, OUTPUT)

print("Done!")
