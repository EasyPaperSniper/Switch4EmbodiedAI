from .configclass import BaseConfig

class SimpleStreamModuleConfig(BaseConfig):
    capture_card_index: int = 0
    viz_stream: bool = True  # Whether to visualize the stream module output
    save_stream: bool = False  # Whether to save the istream module output
    save_path: str = None  # Path to save the stream module output