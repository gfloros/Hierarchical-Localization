import argparse
from pathlib import Path

from hloc import colmap_from_opensfm
from hloc import extract_features
from hloc import logger
from hloc import match_features
from hloc import pairs_from_covisibility


class Dataset:
    def __init__(self, dataset_name: str) -> None:
        self.dataset = Path(f"datasets/{dataset_name}")
        self.outputs = Path(f"outputs/{dataset_name}/")
        self.sfm_pairs = self.outputs / "pairs-sfm.txt"
        self.loc_pairs = self.outputs / "pairs-loc.txt"
        self.sfm_dir = self.outputs / "sfm"
        self.features = self.outputs / "features.h5"
        self.matches = self.outputs / "matches.h5"
        self.feature_conf = extract_features.confs["superpoint_inloc"]
        self.matcher_conf = match_features.confs["superglue"]
        self.references = [
            p.relative_to(self.dataset / "mapping").as_posix()
            for p in (self.dataset / "mapping/").iterdir()
        ]
        self.num_max_covisible_matches = 20

    def do_3d_mapping(self) -> None:
        extract_features.main(
            self.feature_conf,
            self.dataset / "mapping",
            image_list=self.references,
            feature_path=self.features,
        )
        colmap_from_opensfm.main(
            self.dataset / "3D-models",
            self.sfm_dir,
        )
        pairs_from_covisibility.main(
            self.sfm_dir,
            self.sfm_pairs,
            num_matched=self.num_max_covisible_matches,
        )
        match_features.main(
            self.matcher_conf,
            self.sfm_pairs,
            features=self.features,
            matches=self.matches,
        )


def main(dataset_name: str) -> None:
    # Set up dataset
    dataset = Dataset(dataset_name)
    logger.info(f"Dataset: {dataset.dataset}")
    logger.info(f"Outputs: {dataset.outputs}")
    logger.info(f"SfM pairs: {dataset.sfm_pairs}")
    logger.info(f"Loc pairs: {dataset.loc_pairs}")
    logger.info(f"SfM directory: {dataset.sfm_dir}")
    logger.info(f"Features: {dataset.features}")
    logger.info(f"Matches: {dataset.matches}")
    logger.info(f"Feature config: {dataset.feature_conf}")
    logger.info(f"Matcher config: {dataset.matcher_conf}")
    logger.info(f"References: {dataset.references}")

    # 3D mapping
    dataset.do_3d_mapping()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    args = parser.parse_args()
    main(args.dataset)
