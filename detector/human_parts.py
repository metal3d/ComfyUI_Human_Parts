from typing import Tuple

import numpy as np
import torch
from onnxruntime import InferenceSession
from PIL import Image

# classes used in the model
classes = {
    "background": 0,
    "hair": 2,
    "glasses": 4,
    "top-clothes": 5,
    "bottom-clothes": 9,
    "torso-skin": 10,
    "face": 13,
    "left-arm": 14,
    "right-arm": 15,
    "left-leg": 16,
    "right-leg": 17,
    "left-foot": 18,
    "right-foot": 19,
}


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
        if enabled and class_name in classes:
            class_index = classes[class_name]
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
