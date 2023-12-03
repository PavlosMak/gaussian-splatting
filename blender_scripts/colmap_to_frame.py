import struct
import numpy as np
import bpy
import collections

import bmesh

import struct
import numpy as np
import bpy
import collections

import bmesh
from mathutils import Vector
import pandas as pd

Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

EXPERIMENT_DATA_FOLDER = "unity_train_smaller"
OUTPUT = f"/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/train_smaller_frames"

C0 = 0.28209479177387814

def color_to_SH(color: float) -> float:
    return (color - 0.5) / C0

#method adapted from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
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


def select_object_by_name(object_name):
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    # Select the object by name
    bpy.data.objects[object_name].select_set(True)
    # Set the active object
    bpy.context.view_layer.objects.active = bpy.data.objects[object_name]


def save_points_as_frame(object_name: str, output_path: str):
    select_object_by_name(object_name)
    obj = bpy.context.object
    if obj and obj.type == 'MESH':
        print(f"Saving point cloud as optimization animation frame")
        # Iterate over the vertices of the selected mesh
        attributes = obj.data.attributes
        vertices = obj.data.vertices
        entries = []
        for i, vertex in enumerate(vertices):
            i = vertex.index
            xyz = list(vertex.co)
            rgb = [int(channel) for channel in attributes['rgb'].data[i].vector]
            print(rgb)
            row = {
                "id": i,
                "position_x": xyz[0],
                "position_y": xyz[1],
                "position_z": xyz[2],
                "scale_x": 0.1,
                "scale_y": 0.1,
                "scale_z": 0.1,
                "rot_x": 0.0,
                "rot_y": 0.0,
                "rot_z": 0.0,
                "rot_w": 1.0,
                "opacity": 1.0,
                "red_feature": rgb[0],
                "green_feature": rgb[1],
                "blue_feature": rgb[2]
            }
            entries.append(row)
        frame_df = pd.DataFrame(entries)
        frame_df.to_csv(f"{OUTPUT}/0.csv")
    else:
        print("Selected object is not a mesh.")
    print(f"Saved {object_name}")

print("WARNING: This method does not persist the image ids and the 2D point indexes associated with each point.")

OBJECT_NAME = "Point Cloud"
save_points_as_frame(OBJECT_NAME, OUTPUT)

print("Done!")
