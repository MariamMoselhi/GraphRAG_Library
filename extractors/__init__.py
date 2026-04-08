"""
HoloLearn Extractors Package
Maps file extensions and input patterns to extractor types.
"""

try:
    from utils.configs import (
        SUPPORTED_PDF_FORMATS,
        SUPPORTED_DOCX_FORMATS,
        SUPPORTED_PPTX_FORMATS,
        SUPPORTED_VIDEO_FORMATS,
        SUPPORTED_AUDIO_FORMATS,
    )
except ImportError:
    # Fallback defaults if configs cannot be loaded
    SUPPORTED_PDF_FORMATS = ['.pdf']
    SUPPORTED_DOCX_FORMATS = ['.docx', '.doc']
    SUPPORTED_PPTX_FORMATS = ['.pptx', '.ppt']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']

# Extension -> extractor type mapping (built from configs.py lists)
EXTENSION_TYPE_MAP = {}

for ext in SUPPORTED_PDF_FORMATS:
    EXTENSION_TYPE_MAP[ext] = "pdf"

for ext in SUPPORTED_DOCX_FORMATS:
    EXTENSION_TYPE_MAP[ext] = "docx"

for ext in SUPPORTED_PPTX_FORMATS:
    EXTENSION_TYPE_MAP[ext] = "pptx"

for ext in SUPPORTED_VIDEO_FORMATS:
    EXTENSION_TYPE_MAP[ext] = "video"

for ext in SUPPORTED_AUDIO_FORMATS:
    EXTENSION_TYPE_MAP[ext] = "audio"


def get_type_for_extension(ext: str) -> str:
    """
    Get the extractor type for a given file extension.

    Args:
        ext: File extension including dot (e.g. '.pdf')

    Returns:
        Type string ('pdf', 'docx', 'pptx', 'video', 'audio') or 'unknown'
    """
    return EXTENSION_TYPE_MAP.get(ext.lower(), "unknown")
