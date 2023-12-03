import json

import pandas as pd


def read_json_write_csv(json_file, csv_file):
    with open(json_file, 'r') as file:
        cameras_data = json.load(file)
        entries = []
        for camera in cameras_data:
            position = camera["position"]
            rotation = camera["rotation"]
            row = {
                "id": camera["id"],
                "fx": camera["fx"],
                "fy": camera["fy"],
                "height": camera["height"],
                "width": camera["width"],
                "img_name": camera["img_name"],
                "pos_x": position[0],
                "pos_y": position[1],
                "pos_z": position[2],
                "rot_00": rotation[0][0],
                "rot_01": rotation[0][1],
                "rot_02": rotation[0][2],
                "rot_10": rotation[1][0],
                "rot_11": rotation[1][1],
                "rot_12": rotation[1][2],
                "rot_20": rotation[2][0],
                "rot_21": rotation[2][1],
                "rot_22": rotation[2][2]
            }
            entries.append(row)
        camera_df = pd.DataFrame(entries)
        camera_df.to_csv(csv_file)


frame_path = "/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/train_smaller_frames"
camera_path = f"{frame_path}/cameras.json"
output_path = f"{frame_path}/cameras.csv"

read_json_write_csv(camera_path, output_path)
