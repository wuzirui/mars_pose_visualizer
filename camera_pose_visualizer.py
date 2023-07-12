import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image
import tyro
from nerfstudio.configs.base_config import InstantiateConfig, PrintableConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from dataclasses import dataclass, field
from typing import Type
import torch
from torchvision import transforms
from PIL import Image
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class VisConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Runner)
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    image_downsample_factor: int = 5

@dataclass
class Runner:
    config: VisConfig

    def __init__(self, config: VisConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.datamanager = self.config.datamanager.setup()
        self.cameras = self.datamanager.train_dataparser_outputs.cameras
        self.image_filenames = self.datamanager.train_dataparser_outputs.image_filenames
    
    def vis(self) -> None:
        camera_poses = self.cameras.camera_to_worlds
        image_downsample_factor = self.config.image_downsample_factor

        # create a figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # loop over camera poses and plot camera coordinate systems
        for i, camera_pose in enumerate(camera_poses):
            # load and display image
            img_pil = Image.open(self.image_filenames[i])
            if img_pil.mode == "RGBA":
                img_pil = img_pil.convert("RGB")

            # define image transform
            transform = transforms.Compose([
                # transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])


            img_transformed = transform(img_pil).unsqueeze(0)
            img = img_transformed.squeeze().permute((1, 2, 0))
            img = torch.cat([img, 0.5 * torch.ones((*img .shape[:-1], 1))], dim=-1)

            coords = self.cameras.get_image_coords()[::image_downsample_factor, ::image_downsample_factor]
            img = img[::image_downsample_factor, ::image_downsample_factor]
            ray_bundle = self.cameras.generate_rays(camera_indices=i, coords=coords)
            origins = (ray_bundle.origins + ray_bundle.directions).reshape(*img.shape[:-1], 3)
            xx, yy, zz = origins[..., 0], origins[..., 1], origins[..., 2]

            # plot the surface with the image colors
            ax.plot_surface(xx, yy, zz, facecolors=img.numpy(), shade=False)

            # extract camera position and orientation
            camera_position = camera_pose[:3, 3]
            camera_orientation = camera_pose[:3, :3]

            # plot camera coordinate system
            axis_length = 0.8
            x_axis = camera_position + axis_length * camera_orientation[:, 0]
            y_axis = camera_position + axis_length * camera_orientation[:, 1]
            z_axis = camera_position + axis_length * camera_orientation[:, 2]
            ax.plot([camera_position[0], x_axis[0]], [camera_position[1], x_axis[1]], [camera_position[2], x_axis[2]], color='r')
            ax.plot([camera_position[0], y_axis[0]], [camera_position[1], y_axis[1]], [camera_position[2], y_axis[2]], color='g')
            ax.plot([camera_position[0], z_axis[0]], [camera_position[1], z_axis[1]], [camera_position[2], z_axis[2]], color='b')

            # plot frustum boundaries
            corners = origins[[0, -1, 0, -1], [0, 0, -1, -1]]
            for corner in corners:
                ax.plot([camera_position[0], corner[0]], [camera_position[1], corner[1]], [camera_position[2], corner[2]], color=[0.5, 0.5, 0.5, 0.5])

        # set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # set axis limits based on camera positions
        min_pos = torch.min(camera_poses[:, :3, 3], dim=0).values
        max_pos = torch.max(camera_poses[:, :3, 3], dim=0).values
        # ax.set_xlim(min_pos[0], max_pos[0])
        # ax.set_ylim(min_pos[1], max_pos[1])
        # ax.set_zlim(min_pos[2], max_pos[2])

        # show the plot
        plt.show()


def main(config: VisConfig):
    runner = config.setup()
    runner.vis()



if __name__ == "__main__":
    tyro.extras.set_accent_color('bright_yellow')
    main(tyro.cli(VisConfig))
    CONSOLE.log("done")