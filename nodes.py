from typing import Dict, Tuple

import onnxruntime as ort
import torch

from .detector.human_parts import get_mask, labels
from .utils import model_path


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
        def _bool_widget(
            is_enabled=False, tooltip: str | None = None
        ) -> Tuple[str, dict]:
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

        inputs: Dict[str, Dict[str, tuple]] = {
            "required": {
                "image": ("IMAGE",),
            }
        }

        # automate the creation of the inputs using the known labels
        entries: Dict[str, tuple] = {
            segment[0]: _bool_widget(False, f"{segment[1]}")
            for segment in labels.values()
            if segment[0] != ""
        }

        inputs["required"].update(entries)

        return inputs

    def get_mask(self, image: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        """
        Return a Tensor with the mask of the human parts in the image.
        """

        model = ort.InferenceSession(model_path)
        ret_tensor, _ = get_mask(image, model=model, rotation=0, **kwargs)

        return (ret_tensor,)
