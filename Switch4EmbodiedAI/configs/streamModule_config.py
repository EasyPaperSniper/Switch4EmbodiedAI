from .configclass import BaseConfig

class SimpleStreamModuleConfig(BaseConfig):
    capture_card_index: int = 0
    viz_stream: bool = False  # Whether to visualize the image module output
    save_stream: bool = False  # Whether to save the image module output
    save_path: str = None  # Path to save the image module output