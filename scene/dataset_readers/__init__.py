from scene.dataset_readers.old import readColmapSceneInfo, readNerfSyntheticInfo
from scene.dataset_readers.gsbb import readGSBBInfo


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "GSBB": readGSBBInfo,
}