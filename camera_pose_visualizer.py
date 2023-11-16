import numpy as np
import matplotlib.pyplot as plt
import tyro
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from dataclasses import dataclass, field
from typing import Type, Tuple, Optional, List, Dict
import torch
from torchvision import transforms
from PIL import Image
from nerfstudio.utils.rich_utils import CONSOLE

import transforms3d


@dataclass
class VisConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Runner)
    """Target class to instantiate."""
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    """Nerfstudio dataparser config"""
    image_downsample_factor: int = 5
    """the scale to down sample image when visualizing"""
    skip_probability: float = 0.0
    """the probability to skip a datapoint when visualizing"""
    image_plane: float = 1.0
    """the distance of the image plane from the camera center"""
    selected_frames: Optional[Tuple[int, ...]] = None
    """selected frame ids to visualize"""
    show_image: bool = False
    """whether to show the image in the visualization"""
    show_boxes: bool = False
    """whether to show object boxes in the visualization"""


@dataclass
class Runner:
    config: VisConfig

    def __init__(self, config: VisConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.datamanager = self.config.datamanager.setup()
        self.cameras = self.datamanager.train_dataparser_outputs.cameras
        self.image_filenames = self.datamanager.train_dataparser_outputs.image_filenames

    def cuboid_to_3d_points(
        self,
        x: float,
        y: float,
        z: float,
        yaw_angle: float,
        length: float,
        height: float,
        width: float,
    ):
        yaw = yaw_angle
        x = x
        y = y
        z = z
        translation = np.array([[x, y, z]]).T
        # TODO Pierre, be carreful depend on axis UP in global referential (kitti actual version: Y)
        dimension_x = length
        dimension_y = height
        dimension_z = width
        Tr = transforms3d.euler.euler2mat(0, yaw, 0)

        p0 = (
            Tr @ np.array([[dimension_x / 2, dimension_y / 2, dimension_z / 2]]).T
            + translation
        )

        p1 = (
            Tr @ np.array([[-dimension_x / 2, dimension_y / 2, dimension_z / 2]]).T
            + translation
        )

        p2 = (
            Tr @ np.array([[-dimension_x / 2, -dimension_y / 2, dimension_z / 2]]).T
            + translation
        )

        p3 = (
            Tr @ np.array([[dimension_x / 2, -dimension_y / 2, dimension_z / 2]]).T
            + translation
        )

        p4 = (
            Tr @ np.array([[dimension_x / 2, dimension_y / 2, -dimension_z / 2]]).T
            + translation
        )

        p5 = (
            Tr @ np.array([[-dimension_x / 2, dimension_y / 2, -dimension_z / 2]]).T
            + translation
        )

        p6 = (
            Tr @ np.array([[-dimension_x / 2, -dimension_y / 2, -dimension_z / 2]]).T
            + translation
        )

        p7 = (
            Tr @ np.array([[dimension_x / 2, -dimension_y / 2, -dimension_z / 2]]).T
            + translation
        )

        pts = np.hstack((p0, p1, p2, p3, p4, p5, p6, p7)).T

        return pts

    def plot_cuboid(
        self, ax, point3d: np.ndarray, label: bool = None, tracking_id: bool = None
    ):
        point_order_to_plot = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
        ax.plot(
            point3d[point_order_to_plot, 0],
            point3d[point_order_to_plot, 1],
            point3d[point_order_to_plot, 2],
        )
        if label:
            ax.text(point3d[0, 0], point3d[0, 1], point3d[0, 2], label)
        if tracking_id:
            ax.text(point3d[1, 0], point3d[1, 1], point3d[0, 2], tracking_id)

    def vis(self) -> None:
        camera_poses = self.cameras.camera_to_worlds
        image_downsample_factor = self.config.image_downsample_factor

        min_pos = torch.min(camera_poses[:, :3, 3], dim=0).values
        max_pos = torch.max(camera_poses[:, :3, 3], dim=0).values

        # create a figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        skipped = []
        # loop over camera poses and plot camera coordinate systems
        for i, camera_pose in enumerate(camera_poses):
            sample = np.random.rand()
            if (
                sample < self.config.skip_probability
                or (
                    self.config.selected_frames is not None
                    and i not in self.config.selected_frames
                )
                and self.config.show_image
            ):
                skipped.append(i)
                continue
            # [n_frames, n_max_objects, [x,y,z,yaw_angle,track_id, 0]]
            # self.datamanager.train_dataset.metadata['obj_info'][0,0,0,:,:]
            if self.config.show_boxes and (
                "obj_info" not in self.datamanager.train_dataset.metadata
                or "obj_metadata" not in self.datamanager.train_dataset.metadata
            ):
                self.config.show_boxes = False
                print("No object metadata found in this dataset")

            if self.config.show_boxes:
                object_boxes = self.datamanager.train_dataset.metadata["obj_info"][
                    i, 0, 0, :, :
                ]
                object_boxes = object_boxes.reshape(
                    object_boxes.shape[0] // 2, object_boxes.shape[1] * 2
                )
                all_boxes_corners = []
                for object_box in object_boxes:
                    if object_box[0] == -1:
                        continue
                    x, y, z, yaw_angle, track_id_line, _ = object_box
                    (
                        object_id,
                        length,
                        height,
                        width,
                        class_id,
                    ) = self.datamanager.train_dataset.metadata["obj_metadata"][
                        int(track_id_line)
                    ]
                    ax.plot([object_box[0]], [object_box[1]], [object_box[2]], "x")
                    box_corners = self.cuboid_to_3d_points(
                        x, y, z, yaw_angle, length, height, width
                    )

                    self.plot_cuboid(ax, box_corners)
                    all_boxes_corners.append(box_corners)
                all_boxes_corners = np.concatenate(all_boxes_corners)
                min_pos_boxes = np.min(all_boxes_corners[:, :3])
                max_pos_boxes = np.max(all_boxes_corners[:, :3])
            # extract camera position and orientation
            camera_position = camera_pose[:3, 3]
            camera_orientation = camera_pose[:3, :3]

            # plot camera coordinate system
            axis_length = 0.8 * self.config.image_plane
            x_axis = camera_position + axis_length * camera_orientation[:, 0]
            y_axis = camera_position + axis_length * camera_orientation[:, 1]
            z_axis = camera_position + axis_length * camera_orientation[:, 2]
            ax.plot(
                [camera_position[0], x_axis[0]],
                [camera_position[1], x_axis[1]],
                [camera_position[2], x_axis[2]],
                color="r",
            )
            ax.plot(
                [camera_position[0], y_axis[0]],
                [camera_position[1], y_axis[1]],
                [camera_position[2], y_axis[2]],
                color="g",
            )
            ax.plot(
                [camera_position[0], z_axis[0]],
                [camera_position[1], z_axis[1]],
                [camera_position[2], z_axis[2]],
                color="b",
            )
            if self.config.show_image:
                # load and display image
                img_pil = Image.open(self.image_filenames[i])
                if img_pil.mode == "RGBA":
                    img_pil = img_pil.convert("RGB")

                # define image transform
                transform = transforms.Compose(
                    [
                        # transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ]
                )

                img_transformed = transform(img_pil).unsqueeze(0)
                img = img_transformed.squeeze().permute((1, 2, 0))
                img = torch.cat([img, 0.5 * torch.ones((*img.shape[:-1], 1))], dim=-1)

                coords = self.cameras.get_image_coords()[
                    ::image_downsample_factor, ::image_downsample_factor
                ]
                img = img[::image_downsample_factor, ::image_downsample_factor]
                ray_bundle = self.cameras.generate_rays(camera_indices=i, coords=coords)
                origins = (
                    ray_bundle.origins + ray_bundle.directions * self.config.image_plane
                ).reshape(*img.shape[:-1], 3)
                xx, yy, zz = origins[..., 0], origins[..., 1], origins[..., 2]

                # plot the surface with the image colors
                ax.plot_surface(xx, yy, zz, facecolors=img.numpy(), shade=False)

                # plot frustum boundaries
                corners = origins[[0, -1, 0, -1], [0, 0, -1, -1]]
                for corner in corners:
                    ax.plot(
                        [camera_position[0], corner[0]],
                        [camera_position[1], corner[1]],
                        [camera_position[2], corner[2]],
                        color=[0.5, 0.5, 0.5, 0.5],
                    )
                    max_pos = torch.max(max_pos, corner)
                    min_pos = torch.min(min_pos, corner)

            # plot frame id
            ax.text(
                camera_position[0],
                camera_position[1],
                camera_position[2],
                str(i),
                color="black",
                fontsize=10,
            )
        if self.config.show_boxes:
            max_pos = np.max([max_pos.max(), max_pos_boxes]) * 1.1
            min_pos = np.min([min_pos.min(), min_pos_boxes]) * 0.9
        else:
            max_pos = max_pos.max() * 1.1
            min_pos = min_pos.min() * 0.9

        # plot world coordinate axes
        ax.plot([0, 1], [0, 0], [0, 0], color="r")
        ax.plot([0, 0], [0, 1], [0, 0], color="g")
        ax.plot([0, 0], [0, 0], [0, 1], color="b")

        # set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # set axis limits
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylim(min_pos, max_pos)
        ax.set_zlim(min_pos, max_pos)

        CONSOLE.log(f"skipped {len(skipped)} images.")

        # show the plot
        plt.show()


def main(config: VisConfig):
    runner = config.setup()
    runner.vis()


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(VisConfig))
    CONSOLE.log("done")
