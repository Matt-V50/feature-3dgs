
import json
import os
from pathlib import Path

import numpy as np
import torch

from scene.dataset_readers.utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly


def readGSBBInfo(path, foundation_model, eval=True):
    def load_tensor(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    path = Path(path)
    pointcloud="pointcloud.ply"
    transforms = path / "transforms_gsbb.json"
    with open(transforms) as json_file:
        data = json.load(json_file)
    scene_label = data["scene_label"]
    cameras_info = data["json_data"]
    
    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 

    print(f"GSBB Scene:[{scene_label}]")
    
    train_cam_infos = []
    test_cam_infos = []
    
    for idx, transform in enumerate(cameras_info):
        mat = np.array(transform["extrinsic"])
        R = mat[:3,:3]
        T = mat[:3, 3]
        FovY = transform["FovY"]
        FovX = transform["FovX"]
        image_path = transform["image_path"]
        image_name = transform["image_name"]
        width = transform["width"]
        height = transform["height"]
        cx = transform["cx"]
        cy = transform["cy"]
        depth_path = transform["depth_path"]
        train = transform["train"]
            
        image_path = path / image_path
        semantic_feature_path = path / semantic_feature_dir / (image_name + '_fmap_CxHxW.pt')
    
        # image = Image.open(image_path) if image_path.exists() else None
        # object_path = os.path.join(objects_folder, image_name + '.png')
        # objects = Image.open(object_path) if os.path.exists(object_path) else None
    
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                            #   image=image, 
                              image_path=path / image_path,
                              image_name=image_name, width=width, height=height, 
                              semantic_feature_path=semantic_feature_path
                            #   objects=objects
                              )
        if train:
            train_cam_infos.append(cam_info)
        else:
            test_cam_infos.append(cam_info)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        
    train_cam_infos = list(sorted(train_cam_infos, key=lambda x: x.uid))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = path / pointcloud
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        pcd = None
    
    semantic_feature_dim = train_cam_infos[0].semantic_feature.shape[0] 
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim)
    return scene_info