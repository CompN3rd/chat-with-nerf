class Settings:
    data_path: str = "/workspace/chat-with-nerf/data"
    output_path: str = "/workspace/chat-with-nerf/session_output"
    threshold: float = 0.7
    default_scene: str = "office"
    INITIAL_MSG_FOR_DISPLAY = "Hello there! What can I help you find in this room?"
    USE_FAKE_GROUNDER: bool = False
    TYPE_CAPTIONER = "llava"
    LLAVA_PATH = "/workspace/pre-trained-weights/LLaVA/LLaVA-13B-v0"
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
