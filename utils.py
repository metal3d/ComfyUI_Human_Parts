import os

# get the model paths
try:
    from folder_paths import models_dir  # pyright: ignore
except ImportError:
    from pathlib import Path

    models_dir = os.path.join(Path(__file__).parents[2], "models")

models_dir_path = os.path.join(models_dir, "onnx", "human-parts")
model_url = "https://huggingface.co/Metal3d/deeplabv3p-resnet50-human/resolve/main/deeplabv3p-resnet50-human.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir_path, "deeplabv3p-resnet50-human.onnx")
