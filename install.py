import os
import urllib.request

from tqdm import tqdm

try:
    from .utils import model_name, model_path, model_url, models_dir_path
except ImportError:
    from utils import model_name, model_path, model_url, models_dir_path


def download(url, path, name):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get("Content-Length", 0))
    with tqdm(
        total=total,
        desc=f"[HumanParts] Downloading {name} to {path}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        urllib.request.urlretrieve(
            url,
            path,
            reporthook=lambda count, block_size, total_size: progress.update(
                block_size
            ),
        )


if not os.path.exists(models_dir_path):
    os.makedirs(models_dir_path)

if not os.path.exists(model_path):
    download(model_url, model_path, model_name)
