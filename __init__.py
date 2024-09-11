__all__ = ["HumanParts"]

from .nodes import HumanParts

NODE_CLASS_MAPPINGS = {
    "HumanParts": HumanParts,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanParts": "🧍 Human Parts mask generator",
}
