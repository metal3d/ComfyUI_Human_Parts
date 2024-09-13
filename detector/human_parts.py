from typing import Tuple

import numpy as np
import torch
from onnxruntime import InferenceSession
from PIL import Image

# classes used in the model
# Follows CCIHP => https://kalisteo.cea.fr/wp-content/uploads/2021/09/README.html
#
# Note: I prefer to use a dictionnary to be able to change the index if needed.
labels = {
    0: ("background", "Background"),
    1: (
        "hat",
        "Hat: Hat, helmet, cap, hood, veil, headscarf, part covering the skull and hair of a hood/balaclava, crown…",
    ),
    2: (
        "hair",
        "Hair",
    ),
    3: (
        "glove",
        "Glove",
    ),
    4: (
        "glasses",
        "Sunglasses/Glasses: Sunglasses, eyewear, protective glasses…",
    ),
    5: (
        "upper_clothes",
        "UpperClothes: T-shirt, shirt, tank top, sweater under a coat, top of a dress…",
    ),
    6: (
        "face_mask",
        "Face Mask: Protective mask, surgical mask, carnival mask, facial part of a balaclava, visor of a helmet…",
    ),
    7: (
        "coat",
        "Coat: Coat, jacket worn without anything on it, vest with nothing on it, a sweater with nothing on it…",
    ),
    8: (
        "socks",
        "Socks",
    ),
    9: (
        "pants",
        "Pants: Pants, shorts, tights, leggings, swimsuit bottoms… (clothing with 2 legs)",
    ),
    10: (
        "torso-skin",
        "Torso-skin",
    ),
    11: (
        "scarf",
        "Scarf: Scarf, bow tie, tie…",
    ),
    12: (
        "skirt",
        "Skirt: Skirt, kilt, bottom of a dress…",
    ),
    13: (
        "face",
        "Face",
    ),
    14: (
        "left-arm",
        "Left-arm (naked part)",
    ),
    15: (
        "right-arm",
        "Right-arm (naked part)",
    ),
    16: (
        "left-leg",
        "Left-leg (naked part)",
    ),
    17: (
        "right-leg",
        "Right-leg (naked part)",
    ),
    18: (
        "left-shoe",
        "Left-shoe",
    ),
    19: (
        "right-shoe",
        "Right-shoe",
    ),
    20: (
        "bag",
        "Bag: Backpack, shoulder bag, fanny pack… (bag carried on oneself",
    ),
    21: (
        "",
        "Others: Jewelry, tags, bibs, belts, ribbons, pins, head decorations, headphones…",
    ),
}


def get_class_index(class_name: str) -> int:
    """
    Return the index of the class name in the model.
    """
    if class_name == "":
        return -1

    for key, value in labels.items():
        if value[0] == class_name:
            return key

    return -1


def get_mask(
    image: torch.Tensor, model: InferenceSession, rotation: float, **kwargs
) -> Tuple[torch.Tensor, int]:
    """
    Return a Tensor with the mask of the human parts in the image.

    The rotation parameter is not used for now. The idea is to propose rotation to help
    the model to detect the human parts in the image if the character is not in a casual position.
    Several tests have been done, but the model seems to fail to detect the human parts in these cases,
    and the rotation does not help.
    """

    image = image.squeeze(0)
    image_np = image.numpy() * 255

    pil_image = Image.fromarray(image_np.astype(np.uint8))
    original_size = pil_image.size  # to resize the mask later
    # resize to 512x512 as the model expects
    pil_image = pil_image.resize((512, 512))
    center = (256, 256)

    if rotation != 0:
        pil_image = pil_image.rotate(rotation, center=center)

    # normalize the image
    image_np = np.array(pil_image).astype(np.float32) / 127.5 - 1
    image_np = np.expand_dims(image_np, axis=0)

    # use the onnx model to get the mask
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: image_np})
    result = np.array(result[0]).argmax(axis=3).squeeze(0)

    score: int = 0

    mask = np.zeros_like(result)
    for class_name, enabled in kwargs.items():
        class_index = get_class_index(class_name)
        print(f"Class {class_name} is enabled: {enabled} ==> {class_index}")
        if enabled and class_index != -1:
            detected = result == class_index
            mask[detected] = 255
            score += mask.sum()

    # back to the original size
    mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
    if rotation != 0:
        mask_image = mask_image.rotate(-rotation, center=center)

    mask_image = mask_image.resize(original_size)

    # and back to numpy...
    mask = np.array(mask_image).astype(np.float32) / 255

    # add 2 dimensions to match the expected output
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    # ensure to return a "binary mask_image"

    del image_np, result  # free up memory, maybe not necessary
    return (torch.from_numpy(mask.astype(np.uint8)), score)
