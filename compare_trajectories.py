import numpy as np
import matplotlib.pyplot as plt
import tyro
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from dataclasses import dataclass, field
from typing import Type, Tuple, Optional, List, Dict
import torch
from torch import Tensor
from torchvision import transforms
from PIL import Image
from nerfstudio.utils.rich_utils import CONSOLE
from pathlib import Path
import json
import open3d as o3d
import copy


@dataclass
class VisConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Runner)
    """Target class to instantiate."""
    reference_trajectory_path: Path = Path("colmap/transforms.json")
    """path to the reference trajectory"""
    data_trajectory_path: Path = Path("data/transforms.json")
    """path to the data trajectory"""
    image_plane: float = 1.0
    """the distance of the image plane from the camera center"""


@dataclass
class Runner:
    config: VisConfig

    def __init__(self, config: VisConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def read_trajectory_file(self, path: Path) -> Tensor:
        with open(path, "r") as f:
            data = json.load(f)

        frames = sorted(data['frames'], key=lambda x: x['file_path'])
        poses = []
        for frame in frames:
            transform_matrix = frame['transform_matrix']
            pose = np.array(transform_matrix).reshape(4, 4)
            poses.append(pose)
        poses = np.array(poses)
        return torch.from_numpy(poses).float()
    
    def get_camera_frustum_corners(self, camera_pose: Tensor, image_plane: float) -> Tensor:
        """get the four corners of the camera frustum

        Args:
            camera_pose (Tensor): camera to world 4*4 tensor
            image_plane (float): distance of the image plane from the camera center

        Returns:
            Tensor: 4*3 tensor of the four corners of the camera frustum
        """
        pseudo_intrinsics = torch.eye(3).float()
        pseudo_intrinsics[[0, 0], [1, 1]] = 1.0 / image_plane
        pseudo_intrinsics[[0, 1], [2, 2]] = -0.5         # assume a 1*1 image plane
        corner_uv = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)
        corner_camera = torch.matmul(torch.inverse(pseudo_intrinsics), corner_uv.t())
        corner_world = torch.matmul(camera_pose[:3, :3], corner_camera) + camera_pose[:3, 3].unsqueeze(-1)
        return corner_world.t()
    
    def visualize_trajectory(self, ax: plt.Axes, trajectory: Tensor, color: str) -> plt.Axes:
        for i, pose in enumerate(trajectory):
            origin = pose[:3, 3]
            corners = self.get_camera_frustum_corners(pose, self.config.image_plane)

            # plot origin - corner
            ax.plot([origin[0], corners[0, 0]], [origin[1], corners[0, 1]], [origin[2], corners[0, 2]], color=color)
            ax.plot([origin[0], corners[1, 0]], [origin[1], corners[1, 1]], [origin[2], corners[1, 2]], color=color)
            ax.plot([origin[0], corners[2, 0]], [origin[1], corners[2, 1]], [origin[2], corners[2, 2]], color=color)
            
            # plot corner - corner edges
            ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]], [corners[0, 2], corners[1, 2]], color=color)
            ax.plot([corners[1, 0], corners[3, 0]], [corners[1, 1], corners[3, 1]], [corners[1, 2], corners[3, 2]], color=color)
            ax.plot([corners[3, 0], corners[2, 0]], [corners[3, 1], corners[2, 1]], [corners[3, 2], corners[2, 2]], color=color)
            handle = ax.plot([corners[2, 0], corners[0, 0]], [corners[2, 1], corners[0, 1]], [corners[2, 2], corners[0, 2]], color=color)
            
            # # plot origin - origin
            # if i > 0:
            #     ax.plot([trajectory[i-1, 0, 3], origin[0]], [trajectory[i-1, 1, 3], origin[1]], [trajectory[i-1, 2, 3], origin[2]], color=color)
        return handle


    def vis(self) -> None:
        ref_traj = self.read_trajectory_file(self.config.reference_trajectory_path)
        data_traj = self.read_trajectory_file(self.config.data_trajectory_path)

        # apply scale to the reference trajectory
        ref_max_span = (torch.max(ref_traj[:, :3, 3], dim=0).values - torch.min(ref_traj[:, :3, 3], dim=0).values).max()
        data_max_span = (torch.max(data_traj[:, :3, 3], dim=0).values - torch.min(data_traj[:, :3, 3], dim=0).values).max()
        scale = data_max_span / ref_max_span
        ref_traj[:, :3, 3] *= scale

        # ref_mean = torch.mean(ref_traj[:, :3, 3], dim=0)
        # data_mean = torch.mean(data_traj[:, :3, 3], dim=0)
        # ref_traj[:, :3, 3] -= ref_mean
        # data_traj[:, :3, 3] -= data_mean

        # compute 3d-3d ICP
        icp_traj, icp_transform = self.calc_icp(ref_traj, data_traj)
        icp_traj = icp_transform @ data_traj
        
        CONSOLE.log(f"ICP transform:\n {icp_transform}")
        np.savetxt(str(Path("./output") / "icp_transform.txt"), icp_transform.numpy())


        # create a figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        handle_ref = self.visualize_trajectory(ax, ref_traj, "r")
        handle_data = self.visualize_trajectory(ax, data_traj, "b")
        handle_icp = self.visualize_trajectory(ax, icp_traj, "g")

        # get min, max position
        min_pos = min(ref_traj[:, :3, 3].min(), data_traj[:, :3, 3].min())
        max_pos = max(ref_traj[:, :3, 3].max(), data_traj[:, :3, 3].max())
        
        # plot world coordinate axes
        ax.plot([0, max_pos], [0, 0], [0, 0], color="r")
        ax.plot([0, 0], [0, max_pos], [0, 0], color="g")
        ax.plot([0, 0], [0, 0], [0, max_pos], color="b")

        # set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # set axis limits
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylim(min_pos, max_pos)
        ax.set_zlim(min_pos, max_pos)

        # legends
        # red ~ reference trajectory
        fig.legend([handle_ref[0], handle_data[0], handle_icp[0]], ["reference", "data", "icp"], loc="upper left")

        # show the plot
        plt.show()

    def calc_icp(self, ref_traj, data_traj):
        # translate both trajectories
        ref_traj_reg = ref_traj.clone()
        data_traj_reg = data_traj.clone()
        ref_mean = torch.mean(ref_traj[:, :3, 3], dim=0)
        data_mean = torch.mean(data_traj[:, :3, 3], dim=0)
        ref_traj_reg[:, :3, 3] -= ref_mean
        data_traj_reg[:, :3, 3] -= data_mean

        W = torch.zeros(3, 3)
        for i in range(ref_traj.size(0)):
            W += torch.matmul(data_traj_reg[i, :3, 3].unsqueeze(-1), ref_traj_reg[i, :3, 3].unsqueeze(-1).t())
        U, _, V = torch.svd(W)
        R = torch.matmul(U, V.t())
        t = ref_mean - torch.matmul(R, data_mean).squeeze(-1)
        
        # if torch.linalg.det(R) < 0:
        #     R = -R

        icp_transfrom = torch.eye(4)
        icp_transfrom[:3, :3] = R
        icp_transfrom[:3, 3] = t


        icp_traj = torch.matmul(icp_transfrom, data_traj)
        return icp_traj,icp_transfrom


def main(config: VisConfig):
    runner = config.setup()
    runner.vis()


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(VisConfig))
    CONSOLE.log("done")
