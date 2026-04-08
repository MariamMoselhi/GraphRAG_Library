import os
from pathlib import Path

# Import GPU auto-detection (will automatically detect system capabilities)
try:
    from utils.gpu_detector import get_auto_config, get_gpu_detector
    _AUTO_CONFIG = get_auto_config()
    _GPU_DETECTOR = get_gpu_detector()
except ImportError:
    # Fallback if gpu_detector not available yet
    _AUTO_CONFIG = {
        "device": "cpu",
        "easyocr_gpu": True,
        "easyocr_batch_size": 4,
        "pix2tex_device": "cpu",
        "pix2tex_batch_size": 2,
        "use_gpu_for_video": True,
        "frame_extraction_batch_size": 4,
    }
    _GPU_DETECTOR = None

# ==================== API CONFIGURATION ====================
# Groq API Key (Set this in your environment or replace with your key)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY_VIDEO = os.getenv("GROQ_API_KEY_VIDEO")
GROQ_API_KEY_AUDIO = os.getenv("GROQ_API_KEY_AUDIO")


# Groq Models
WHISPER_MODEL = "whisper-large-v3-turbo"  # For audio transcription
# Alternative: "whisper-large-v3-turbo" for faster processing

# ==================== PATHS CONFIGURATION ====================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"  # For temporary files during processing

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ==================== OCR CONFIGURATION ====================
# Auto-detected settings based on GPU availability and compatibility
# These values are automatically configured by gpu_detector.py
EASYOCR_LANGUAGES = ['en']  # Add more languages as needed: ['en', 'ar']
# GPU for EasyOCR: auto-detected from system. Set FORCE_GPU=true in .env to override.
_FORCE_GPU = os.getenv("FORCE_GPU", "").lower() in ("true", "1", "yes")
EASYOCR_GPU = _FORCE_GPU or _AUTO_CONFIG.get("easyocr_gpu", False)
EASYOCR_BATCH_SIZE = _AUTO_CONFIG.get("easyocr_batch_size", 4)  # Auto-optimized

# ==================== LATEX-OCR CONFIGURATION (Mathematical Equations OCR) ====================
# LaTeX-OCR (pix2tex) Settings - Specialized equation recognition
LATEX_OCR_DEVICE = _AUTO_CONFIG.get("pix2tex_device", "cpu")  # Auto-detected
LATEX_OCR_AVAILABLE = True  # Will be checked at runtime

# Smart Math Detection - Only use LaTeX-OCR on frames with mathematical content
ENABLE_SMART_MATH_DETECTION = True  # Set to False to run on all frames
MATH_SYMBOLS = ['=', '+', '-', '×', '÷', '∫', '∑', '∂', '∆', 'α', 'β', 'γ', 'θ', 'λ', 'μ', 'σ', 
                '√', '∞', '≤', '≥', '≠', '≈', '∈', '∉', '⊂', '⊃', '∪', '∩', '^', '_', '²', '³']

# Pix2Text Settings (for mathematical equations) - Auto-detected
PIX2TEX_MODEL = "formula_recognition"  # LaTeX OCR model
PIX2TEX_DEVICE = _AUTO_CONFIG.get("pix2tex_device", "cpu")  # Auto-detected
PIX2TEX_BATCH_SIZE = _AUTO_CONFIG.get("pix2tex_batch_size", 2)  # Auto-optimized

# ==================== VIDEO PROCESSING CONFIGURATION ====================
# Frame Extraction Settings - Auto-detected based on GPU compatibility
FRAME_EXTRACTION_FPS = 0.5  # Extract 1 frame per second
FRAME_EXTRACTION_BATCH_SIZE = _AUTO_CONFIG.get("frame_extraction_batch_size", 4)  # Auto-optimized
SCENE_CHANGE_THRESHOLD = 30.0  # Threshold for detecting scene changes (0-100)
MAX_FRAMES_PER_VIDEO = 1000  # Limit frames to prevent excessive processing

# Video Format Support
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# GPU Acceleration for Video Processing - Auto-detected
USE_GPU_FOR_FRAME_EXTRACTION = _AUTO_CONFIG.get("use_gpu_for_video", False)  # Auto-detected

# ==================== AUDIO PROCESSING CONFIGURATION ====================
# Audio Format Support
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']

# Whisper Settings
WHISPER_LANGUAGE = "en"  # Set to None for auto-detection
WHISPER_TEMPERATURE = 0.0  # Lower = more focused, higher = more creative

# ==================== DOCUMENT PROCESSING CONFIGURATION ====================
# Supported Document Formats
SUPPORTED_PDF_FORMATS = ['.pdf']
SUPPORTED_PPTX_FORMATS = ['.pptx', '.ppt']
SUPPORTED_DOCX_FORMATS = ['.docx', '.doc']

# ==================== URL SCRAPING CONFIGURATION ====================
# Request Settings
REQUEST_TIMEOUT = 30  # Timeout in seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Retry Settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
BACKOFF_FACTOR = 2  # Exponential backoff multiplier

# ==================== OUTPUT CONFIGURATION ====================
# Output File Settings
OUTPUT_FILENAME = "extracted_content.txt"
INCLUDE_TIMESTAMPS = True
INCLUDE_SOURCE_MARKERS = True

# Source Marker Format
SOURCE_MARKER_FORMAT = "\n{'='*80}\nSOURCE: {filename}\nTYPE: {filetype}\nTIMESTAMP: {timestamp}\n{'='*80}\n"

# ==================== ERROR HANDLING CONFIGURATION ====================
# Logging Settings
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILENAME = "extractor.log"

# Error Handling
CONTINUE_ON_ERROR = True  # Continue processing other files if one fails
SAVE_PARTIAL_RESULTS = True  # Save successfully extracted content even if some fail

# ==================== PROCESSING LIMITS ====================
# File Size Limits (in MB)
MAX_PDF_SIZE = 100
MAX_VIDEO_SIZE = 500
MAX_AUDIO_SIZE = 200
MAX_IMAGE_SIZE = 50

# Processing Timeouts (in seconds)
PDF_TIMEOUT = 300
VIDEO_TIMEOUT = 600
AUDIO_TIMEOUT = 300
URL_TIMEOUT = 60

# ==================== FEATURE FLAGS ====================
ENABLE_MATH_OCR = False  # Enable pix2tex (LaTeX-OCR) for visual math equations in frames
ENABLE_SPOKEN_MATH_DETECTION = False  # Detect math in Whisper audio transcripts and convert to LaTeX
ENABLE_SCENE_DETECTION = True  # Use scene detection for video frame extraction
ENABLE_DEDUPLICATION = True  # Remove duplicate extracted text
ENABLE_TEXT_CLEANING = True  # Clean and normalize extracted text

# ==================== MODERATOR CONFIGURATION ====================
MAX_PARALLEL_EXTRACTIONS = 4  # Thread pool size for parallel extraction

# ==================== OPTIONAL: GROQ RATE LIMITING ====================
# Groq Free Tier: 30 requests/minute, 14,400 requests/day
GROQ_RATE_LIMIT_RPM = 30
GROQ_RATE_LIMIT_RPD = 14400

# ==================== CONFIGURATION STATUS ====================
def print_config_status():
    """Print current configuration status with GPU detection info"""
    print("\n" + "="*80)
    print("EXTRACTOR CONFIGURATION STATUS")
    print("="*80)
    
    if _GPU_DETECTOR:
        print("\nGPU DETECTION:")
        print(f"  Device: {_AUTO_CONFIG.get('device', 'cpu').upper()}")
        print(f"  GPU Name: {_GPU_DETECTOR.gpu_name}")
        print(f"  CUDA Available: {_GPU_DETECTOR.cuda_available}")
        print(f"  GPU Compatible: {_GPU_DETECTOR.gpu_compatible}")
        if _GPU_DETECTOR.cuda_available:
            print(f"  VRAM: {_GPU_DETECTOR.vram_gb:.1f} GB")
        print(f"  Status: {_GPU_DETECTOR.compatibility_message}")
    
    print("\nOCR SETTINGS:")
    print(f"  EasyOCR GPU: {EASYOCR_GPU}")
    print(f"  EasyOCR Batch Size: {EASYOCR_BATCH_SIZE}")
    print(f"  Pix2Text Device: {PIX2TEX_DEVICE}")
    print(f"  Pix2Text Batch Size: {PIX2TEX_BATCH_SIZE}")
    
    print("\nVIDEO PROCESSING:")
    print(f"  GPU Acceleration: {USE_GPU_FOR_FRAME_EXTRACTION}")
    print(f"  Frame Batch Size: {FRAME_EXTRACTION_BATCH_SIZE}")
    print(f"  FPS: {FRAME_EXTRACTION_FPS}")
    
    print("\nFILE SUPPORT:")
    print(f"  Video: {', '.join(SUPPORTED_VIDEO_FORMATS)}")
    print(f"  Audio: {', '.join(SUPPORTED_AUDIO_FORMATS)}")
    print(f"  PDF: {', '.join(SUPPORTED_PDF_FORMATS)}")
    
    print("\nAPI:")
    print(f"  Groq API Key: {'✓ Set' if GROQ_API_KEY != 'your-groq-api-key-here' else '✗ Not Set'}")
    print(f"  Whisper Model: {WHISPER_MODEL}")
    
    print("="*80 + "\n")

# Show config on import (optional - comment out if not needed)
if __name__ != "__main__":
    pass  # Silent on import, call print_config_status() manually if needed