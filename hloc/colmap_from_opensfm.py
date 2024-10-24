import argparse
import collections
import csv
import json
import math
from pathlib import Path
from typing import TypeAlias

import numpy as np
from tqdm import tqdm

from hloc import logger
from hloc.utils.read_write_model import (
    CAMERA_MODEL_NAMES,
    Camera,
    Image,
    Point3D,
    write_model,
)

FeaturePoint = collections.namedtuple("FeaturePoint", ["id", "x", "y"])
FeaturePointRef = collections.namedtuple("FeaturePointRef", ["shot_id", "track_id"])
ImagePointsMap: TypeAlias = dict[str, list[FeaturePoint]]
Points2dMap: TypeAlias = dict[int, list[FeaturePointRef]]
CamerasMap: TypeAlias = dict[str, Camera]
Points3dMap: TypeAlias = dict[int, Point3D]
ImagesMap: TypeAlias = dict[int, Image]
StrIdsMap: TypeAlias = dict[str, int]
IntIdsMap: TypeAlias = dict[int, int]


def angle_axis_to_quaternion(angle_axis: np.ndarray) -> list[float]:
    angle = np.linalg.norm(angle_axis)
    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle
    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)
    return np.array([qw, float(qx), float(qy), float(qz)])


def denormalize_image_coordinates(
    norm_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p


def read_tracks(opensfm_path: str) -> tuple[ImagePointsMap, Points2dMap]:
    logger.info("Reading OpenSfM tracks...")
    images_points_map = ImagePointsMap()
    points_map = Points2dMap()
    with open(opensfm_path / "tracks.csv", "r") as fin:
        tracks_reader = csv.reader(fin, delimiter="\t")
        next(tracks_reader, None)  # skip the header
        for row in tracks_reader:
            shot_id = row[0]
            track_id = int(row[1])
            feature_id = int(row[2])
            x = float(row[3])
            y = float(row[4])
            img = images_points_map.get(shot_id, None)
            if img is None:
                images_points_map[shot_id] = list()
            feature_point = FeaturePoint(feature_id, x, y)
            images_points_map[shot_id].append(feature_point)
            point = points_map.get(feature_id, None)
            if point is None:
                points_map[feature_id] = list()
            feature_point_ref = FeaturePointRef(shot_id, track_id)
            points_map[feature_id].append(feature_point_ref)
    return images_points_map, points_map


def read_cameras(reconstruction_json: dict) -> tuple[CamerasMap, StrIdsMap]:
    camera_ids_map = StrIdsMap()
    cameras = CamerasMap()
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
            params=[],
        )
        cameras[camera_id] = camera
    return cameras, camera_ids_map


def sniff_images(reconstruction_json: dict) -> StrIdsMap:
    image_ids_map = StrIdsMap()
    for idx, key in enumerate(reconstruction_json["shots"].keys()):
        image_ids_map[key] = idx
    return image_ids_map


def read_points(
    reconstruction_json: dict,
    points_map: Points2dMap,
    image_ids_map: StrIdsMap,
) -> tuple[Points3dMap, IntIdsMap]:
    point3d_ids_map = IntIdsMap()
    points3d = Points3dMap()
    num_points = len(reconstruction_json["points"])
    pbar = tqdm(total=num_points, unit="pts")
    for idx, (key, value) in enumerate(reconstruction_json["points"].items()):
        point_id = int(key)
        point3d_ids_map[point_id] = idx
        coordinates = value["coordinates"]
        color = value["color"]
        xyz = np.array([coordinates[0], coordinates[1], coordinates[2]], float)
        color_arr = np.array([color[0], color[1], color[2]], int)
        points = points_map.get(point_id, [])
        image_ids = np.array([image_ids_map[p.shot_id] for p in points], int)
        points2d_idxs = np.array([p.track_id for p in points], int)
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
    return points3d, point3d_ids_map


def read_images(
    reconstruction_json: dict,
    camera_ids_map: StrIdsMap,
    image_ids_map: StrIdsMap,
    image_points_map: ImagePointsMap,
    point3d_ids_map: IntIdsMap,
    cameras_map: CamerasMap,
) -> ImagesMap:
    images = ImagesMap()
    for key, value in reconstruction_json["shots"].items():
        camera_id = camera_ids_map[value["camera"]]
        image_id = image_ids_map[key]
        image_name = key
        rvec = value["rotation"]
        tvec = value["translation"]
        qvec = angle_axis_to_quaternion(rvec)
        tvec_arr = np.array([tvec[0], tvec[1], tvec[2]])
        images_points = image_points_map.get(key, [])
        norm_xys = np.array(
            [[p.x, p.y] for p in images_points if point3d_ids_map.get(p.id, None)],
            float,
        )
        camera = cameras_map[camera_id]
        xys = denormalize_image_coordinates(norm_xys, camera.width, camera.height)
        point3d_ids = np.array(
            [
                point3d_ids_map[p.id]
                for p in images_points
                if point3d_ids_map.get(p.id, None)
            ],
            int,
        )
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
    return images


def read_opensfm_model(
    opensfm_path: str,
    images_points_map: ImagePointsMap,
    points_map: Points2dMap,
) -> tuple[CamerasMap, ImagesMap, Points3dMap]:
    logger.info("Reading OpenSfM reconstruction...")
    with open(opensfm_path / "reconstruction.json", "r") as fin:
        reconstructions_json = json.load(fin)
    assert len(reconstructions_json) == 1
    reconstruction_json = reconstructions_json[0]
    logger.info("Reading cameras...")
    cameras, camera_ids_map = read_cameras(reconstruction_json)
    logger.info("Sniffing images...")
    image_ids_map = sniff_images(reconstruction_json)
    logger.info("Reading points...")
    points3d, point3d_ids_map = read_points(
        reconstruction_json,
        points_map,
        image_ids_map,
    )
    logger.info("Reading images...")
    images = read_images(
        reconstruction_json,
        camera_ids_map,
        image_ids_map,
        images_points_map,
        point3d_ids_map,
        cameras,
    )
    return cameras, images, points3d


def main(opensfm_path: str, output: str) -> None:
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
