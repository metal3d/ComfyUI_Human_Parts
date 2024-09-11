from typing import Tuple

import onnxruntime as ort
import torch

from .utils import model_path
from .detector.human_parts import get_mask


class HumanParts:
    """
    This node is used to get a mask of the human parts in the image.

    The model used is DeepLabV3+ with a ResNet50 backbone trained
    by Keras-io, converted to ONNX format.

    """

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "get_mask"
    CATEGORY = "Metal3d"

    @classmethod
    def INPUT_TYPES(cls):
        def _bool_widget(is_enabled=False, tooltip: str | None = None):
            """Helper function to create a boolean widget"""
            return (
                "BOOLEAN",
                {
                    "default": is_enabled,
                    "label_on": "Enabled",
                    "label_off": "Disabled",
                    "tooltip": tooltip,
                },
            )

        return {
            "required": {
                "image": ("IMAGE",),
                "background": _bool_widget(
                    tooltip="Background, excluding human parts, invert this mask to get the human parts",
                ),
                "face": _bool_widget(
                    is_enabled=True,
                    tooltip="Face, including eyes, mouth, etc.",
                ),
                "hair": _bool_widget(
                    tooltip="Hair, including beard, mustache, etc.",
                ),
                "glasses": _bool_widget(
                    tooltip="Glasses, sunglasses, etc. Eyes can be included"
                ),
                "top-clothes": _bool_widget(
                    tooltip="Shirt, T-shirt, etc.",
                ),
                "bottom-clothes": _bool_widget(
                    tooltip="Pants, shorts, etc.",
                ),
                "torso-skin": _bool_widget(
                    tooltip="Skin of the torso, excluding clothes. Neck can be included"
                ),
                "left-arm": _bool_widget(
                    tooltip="Left arm, excluding clothes, hand can be included"
                ),
                "right-arm": _bool_widget(
                    tooltip="Right arm, excluding clothes, hand can be included"
                ),
                "left-leg": _bool_widget(
                    tooltip="Left leg, excluding clothes, foot can be included"
                ),
                "right-leg": _bool_widget(
                    tooltip="Right leg, excluding clothes, foot can be included"
                ),
                "left-foot": _bool_widget(
                    tooltip="Left foot, excluding shoes",
                ),
                "right-foot": _bool_widget(
                    tooltip="Right foot, excluding shoes",
                ),
            }
        }

    def get_mask(self, image: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        """
        Return a Tensor with the mask of the human parts in the image.
        """

        model = ort.InferenceSession(model_path)
        ret_tensor, _ = get_mask(image, model=model, rotation=0, **kwargs)

        return (ret_tensor,)
