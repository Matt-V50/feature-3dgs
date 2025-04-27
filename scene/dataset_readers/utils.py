from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2
import numpy as np
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
import torch

def fetchPly(path, only_xyz=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors, normals = None, None
    names = list(map(lambda k:k.name, vertices.properties))
    if not only_xyz:
        if 'red' not in names:
            colors = np.ones_like(positions)
        else:
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        # check if nx is one property of the verticess
        if 'nx' not in names:
            normals = np.zeros_like(positions)
        else:
            np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic_feature_path: Path
    
    _semantic_feature: torch.tensor = None
    _image: Image = None
    
    @property
    def image(self) -> Image:
        """
        lazy load image
        """
        if self._image is None:
            self._image = Image.open(self.image_path)
    
        return self._image
    
    @property
    def semantic_feature(self) -> torch.tensor:
        """
        lazy load semantic feature
        """
        if self._semantic_feature is None:
            self._semantic_feature = torch.load(self.semantic_feature_path, map_location="cpu", weights_only=True)
    
        return self._semantic_feature


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    semantic_feature_dim: int 

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}