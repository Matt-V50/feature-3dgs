conda install nvidia/label/cuda-12.1.1::cuda-toolkit
conda install pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH pip install git+https://github.com/nerfstudio-project/gsplat.git@v0.1.10

pip install plyfile opencv-python imageio \
ftfy regex tqdm \
git+https://github.com/openai/CLIP.git \
git+https://github.com/zhanghang1989/PyTorch-Encoding/ \
altair \
streamlit  \
protobuf \
pytorch_lightning \
timm \
tensorboard \
tensorboardX \
matplotlib \
test-tube \
wandb \
torchmetrics \
scikit-image \
scikit-learn \
pycocotools \
onnxruntime  \
onnx \
submodules/simple-knn