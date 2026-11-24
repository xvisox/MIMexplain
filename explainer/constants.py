from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
EXAMPLES_DIR = ASSETS_DIR / "examples"

DEFAULT_MODEL_NAME = "facebook/metaclip-2-worldwide-b16"
DEFAULT_TEXT_LAYERS = 8

# 512x512 produces a 32x32 patch grid for the b16 MetaCLIP-2 vision encoder.
DEFAULT_RESIZED_IMAGE_SIZE = 32 * 16
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metaclip2"
