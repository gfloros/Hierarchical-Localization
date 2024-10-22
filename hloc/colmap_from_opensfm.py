import argparse
import csv
import json
import math
import typing

from pathlib import Path
from tqdm import tqdm

import numpy as np

from hloc import logger
from hloc.utils.read_write_model import (
    CAMERA_MODEL_NAMES,
    Camera,
    Image,
    Point3D,
    write_model
)


def angle_axis_to_quaternion(angle_axis: np.ndarray) -> typing.List[float]:
    angle = np.linalg.norm(angle_axis)
    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle
    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)
    return np.array([qw, float(qx), float(qy), float(qz)])


def read_tracks(opensfm_path):
    logger.info("Reading OpenSfM tracks...")
    images_points_map = {}
    points_map = {}
    with open(opensfm_path / "tracks.csv", "r") as fin:
        tracks_reader = csv.reader(fin, delimiter="\t")
        next(tracks_reader, None)  # skip the header
        for row in tracks_reader:
            shot_id = row[0]
            track_id = int(row[1])
            feature_id = int(row[2])
            x = float(row[3])
            y = float(row[4])
            scale = float(row[5])
            r = int(row[6])
            g = int(row[7])
            b = int(row[8])
            img = images_points_map.get(shot_id, None)
            if img is None:
                images_points_map[shot_id] = list()
            images_points_map[shot_id].append({"x": x, "y": y, "feature_id": feature_id})
            point = points_map.get(feature_id, None)
            if point is None:
                points_map[feature_id] = list()
            points_map[feature_id].append({"shot_id": shot_id, "track_id": track_id})
    return images_points_map, points_map
        

def read_opensfm_model(opensfm_path, images_points_map, points_map):
    logger.info("Reading OpenSfM reconstruction...")
    with open(opensfm_path / "reconstruction.json", "r") as fin:
        reconstructions_json = json.load(fin)
    assert len(reconstructions_json) == 1
    reconstruction_json = reconstructions_json[0]
    logger.info("Reading cameras...")
    camera_ids_map = {}
    cameras = {}
    for idx, (key, value) in enumerate(reconstruction_json["cameras"].items()):
        projection_type = value["projection_type"]
        assert projection_type == "spherical"
        camera_model = CAMERA_MODEL_NAMES["SPHERICAL"]
        camera_id = idx
        camera_ids_map[key] = camera_id
        width = int(value.get("width", 0))
        height = int(value.get("height", 0))
        camera = Camera(
            id=camera_id,
            model=camera_model.model_name,
            width=width,
            height=height,
            params=[]
        )
        cameras[camera_id] = camera
    image_ids_map = {}
    for idx, key in enumerate(reconstruction_json["shots"].keys()):
        image_ids_map[key] = idx
    logger.info("Reading points...")
    point3d_ids_map = {}
    points3d = {}
    num_points = len(reconstruction_json["points"])
    pbar = tqdm(total=num_points, unit="pts")
    for idx, (key, value) in enumerate(reconstruction_json["points"].items()):
        point3d_ids_map[int(key)] = idx
        coordinates = value["coordinates"]
        color = value["color"]
        xyz = np.array([coordinates[0], coordinates[1], coordinates[2]], float)
        color_arr = np.array([color[0], color[1], color[2]], int)
        points = points_map.get(key, [])
        image_ids = np.array([image_ids_map[p["shot_id"]] for p in points], int)
        points2d_idxs = np.array([p["track_id"] for p in points], int)
        point = Point3D(
            id=idx,
            xyz=xyz,
            rgb=color_arr,
            error=1.0,  # fake
            image_ids=image_ids,
            point2D_idxs=points2d_idxs,
        )
        points3d[idx] = point
        pbar.update(1)
    pbar.close()
    logger.info("Reading images...")
    images = {}
    for key, value in reconstruction_json["shots"].items():
        camera_id = camera_ids_map[value["camera"]]
        image_id = image_ids_map[key]
        image_name = key
        rvec = value["rotation"]
        tvec = value["translation"]
        qvec = angle_axis_to_quaternion(rvec)
        tvec_arr = np.array([tvec[0], tvec[1], tvec[2]])
        images_points = images_points_map.get(key, [])
        xys = np.array([[p["x"], p["y"]] for p in images_points if point3d_ids_map.get(p["feature_id"], None)], float)
        point3d_ids = np.array([point3d_ids_map[p["feature_id"]] for p in images_points if point3d_ids_map.get(p["feature_id"], None)], int)
        image = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec_arr,
            camera_id=camera_id,
            name=image_name,
            xys=xys,
            point3D_ids=point3d_ids,
        )
        images[image_id] = image
    return cameras, images, points3d


def main(opensfm_path, output):
    assert opensfm_path.exists(), opensfm_path
    logger.info("Reading the OpenSfM tracks...")
    images_points_map, points_map = read_tracks(opensfm_path)
    
    logger.info("Reading the OpenSfM model...")
    model = read_opensfm_model(opensfm_path, images_points_map, points_map)

    logger.info("Writing the COLMAP model...")
    output.mkdir(exist_ok=True, parents=True)
    write_model(*model, path=str(output), ext=".bin")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensfm-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.opensfm_path, args.output)
