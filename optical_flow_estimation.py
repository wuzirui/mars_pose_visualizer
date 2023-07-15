import numpy as np
import matplotlib.pyplot as plt
import tyro
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from dataclasses import dataclass, field
from typing import Type, Dict
import torch
from torchvision import transforms
from PIL import Image
from nerfstudio.utils.rich_utils import CONSOLE
import cv2
from pathlib import Path


@dataclass
class VisConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Runner)
    """Target class to instantiate."""
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    """Nerfstudio dataparser config"""
    output_dir: Path = Path("output/optical_flow")
    """output directory"""
    corner_detection_parameters: Dict = field(default_factory=lambda: {"maxCorners": 100, "qualityLevel": 0.3, "minDistance": 7, "blockSize": 7})
    """parameters for corner detection"""
    lk_flow_paramsters: Dict = field(default_factory=lambda: {"winSize": (15, 15), "maxLevel": 2, "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)})
    """parameters for Lucas-Kanade optical flow"""


@dataclass
class Runner:
    config: VisConfig

    def __init__(self, config: VisConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.datamanager = self.config.datamanager.setup()
        self.cameras = self.datamanager.train_dataparser_outputs.cameras
        self.image_filenames = self.datamanager.train_dataparser_outputs.image_filenames
        self.config.output_dir.mkdir(exist_ok=True, parents=True)

    def vis(self) -> None:
        camera_poses = self.cameras.camera_to_worlds
        featureParams = self.config.corner_detection_parameters
        lkParams = self.config.lk_flow_paramsters

        img_ref = cv2.imread(str(self.image_filenames[0]))
        # extract good features of the first reference frame
        img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        ref_points = cv2.goodFeaturesToTrack(img_ref_gray, **featureParams)

        for new_image_filename in self.image_filenames[1:]:
            img_new = cv2.imread(str(new_image_filename))
            img_new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

            new_points = cv2.goodFeaturesToTrack(img_new_gray, **featureParams)
            
            new_points, status, error = cv2.calcOpticalFlowPyrLK(img_ref_gray, img_new_gray, ref_points, new_points, **lkParams)

            # select good points
            good_new, good_old = new_points[status == 1], ref_points[status == 1]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                output = cv2.line(img_new, (a, b), (c, d), (0, 255, 0), 2)
                output = cv2.circle(img_new, (a, b), 5, (0, 0, 255), -1)
            cv2.imwrite(str(self.config.output_dir / f"{new_image_filename.stem}_flow.png"), output)

            # update the previous frame and previous points
            img_ref = img_new.copy()
            img_ref_gray = img_new_gray.copy()
            ref_points = good_new.reshape(-1, 1, 2)


def main(config: VisConfig):
    runner = config.setup()
    runner.vis()


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(VisConfig))
    CONSOLE.log("done")
