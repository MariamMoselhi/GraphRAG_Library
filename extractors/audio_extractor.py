"""
Audio Extractor for HoloLearn
Extracts text from audio files using Groq Whisper API.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import time
import re

# Import our utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.configs import (
    OUTPUT_DIR, 
    LOGS_DIR,
    GROQ_API_KEY_AUDIO,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_TEMPERATURE,
    SUPPORTED_AUDIO_FORMATS,
    MAX_AUDIO_SIZE
)
from utils.error_handler import ErrorHandler
from utils.text_cleaner import TextCleaner

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


class AudioExtractor:
    """Extract text from audio files using Groq Whisper"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Audio Extractor
        
        Args:
            api_key: Groq API key (if None, uses config)
        """
        self.text_cleaner = TextCleaner()
        self.base_output_dir = OUTPUT_DIR
        self.base_logs_dir = LOGS_DIR
        
        # Ensure base directories exist
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Groq client
        if not _GROQ_AVAILABLE:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )

        self.api_key = api_key or GROQ_API_KEY_AUDIO

        if not self.api_key or self.api_key == "your-groq-api-key-here":
            raise ValueError(
                "Groq API key not set! Please set GROQ_API_KEY in your .env file or config.py"
            )

        self.client = Groq(api_key=self.api_key)
    
    def _create_resource_name(self, filename: str) -> str:
        """
        Create a clean resource name from filename
        
        Example: "Lecture Recording - Week 5.mp3" → "lecture_recording_week_5"
        """
        # Remove extension
        name = Path(filename).stem
        
        # Convert to lowercase
        name = name.lower()
        
        # Replace spaces and special chars with underscore
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Limit length
        if len(name) > 50:
            name = name[:50]
        
        return name or "unnamed_audio"
    
    def _setup_resource_directories(self, resource_name: str, output_dir_override: Optional[Path] = None) -> tuple:
        """
        Create directories for a specific resource

        Args:
            resource_name: Clean name derived from the input filename.
            output_dir_override: If provided, use this directory for output
                instead of creating a resource-specific subfolder.

        Returns:
            (output_dir, logs_dir) paths
        """
        if output_dir_override:
            resource_output_dir = Path(output_dir_override)
        else:
            resource_output_dir = self.base_output_dir / resource_name

        resource_logs_dir = self.base_logs_dir / resource_name

        resource_output_dir.mkdir(parents=True, exist_ok=True)
        resource_logs_dir.mkdir(parents=True, exist_ok=True)

        return resource_output_dir, resource_logs_dir
    
    def _validate_audio_file(self, audio_path: Path, error_handler: ErrorHandler) -> bool:
        """
        Validate audio file
        
        Returns:
            True if valid, False otherwise
        """
        # Check if file exists
        if not audio_path.exists():
            error_handler.log_error(
                FileNotFoundError(f"Audio file not found: {audio_path}"),
                context="Validating audio file"
            )
            return False
        
        # Check file extension
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            error_handler.log_error(
                ValueError(f"Unsupported audio format: {audio_path.suffix}"),
                context="Validating audio format",
                metadata={"supported": SUPPORTED_AUDIO_FORMATS}
            )
            return False
        
        # Check file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_AUDIO_SIZE:
            error_handler.log_error(
                ValueError(f"Audio file too large: {file_size_mb:.2f}MB (max: {MAX_AUDIO_SIZE}MB)"),
                context="Checking audio file size"
            )
            return False
        
        return True
    
    def _transcribe_with_whisper(self, 
                                 audio_path: Path,
                                 error_handler: ErrorHandler) -> Optional[str]:
        """
        Transcribe audio using Groq Whisper API
        
        Returns:
            Transcription text, or None if failed
        """
        try:
            error_handler.log_info(
                f"Transcribing audio with Whisper...",
                metadata={
                    "model": WHISPER_MODEL,
                    "language": WHISPER_LANGUAGE or "auto-detect"
                }
            )
            
            # Open audio file
            with open(audio_path, "rb") as audio_file:
                # Call Groq Whisper API
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file.read()),
                    model=WHISPER_MODEL,
                    language=WHISPER_LANGUAGE,  # None for auto-detect
                    temperature=WHISPER_TEMPERATURE,
                    response_format="verbose_json"  # Get detailed response
                )
            
            # Extract text from response
            text = transcription.text
            
            # Get additional metadata if available
            language = getattr(transcription, 'language', 'unknown')
            duration = getattr(transcription, 'duration', 0)
            
            error_handler.log_success(
                "Audio transcribed successfully",
                metadata={
                    "language": language,
                    "duration": f"{duration:.2f}s" if duration else "unknown",
                    "length": len(text)
                }
            )
            
            return text
            
        except Exception as e:
            error_handler.log_error(
                e,
                context="Transcribing audio with Whisper API",
                metadata={"audio_file": audio_path.name}
            )
            return None
    
    def extract(self,
                audio_path: str,
                resource_id: Optional[str] = None,
                clean_text: bool = True,
                language: Optional[str] = None,
                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from an audio file

        Args:
            audio_path: Path to audio file
            resource_id: Optional unique identifier (if None, uses filename)
            clean_text: Whether to clean the transcribed text
            language: Language code (e.g., 'en', 'ar') or None for auto-detect
            output_dir: Optional shared output directory. When provided,
                all output files are written here instead of a per-resource subfolder.

        Returns:
            Dictionary with extraction results and metadata

        Example:
            extractor = AudioExtractor()
            result = extractor.extract("lecture.mp3")
            # Creates: output/lecture/lecture_text.txt and output/lecture/lecture_metadata.json
        """
        start_time = time.time()
        audio_path = Path(audio_path)

        # Create resource name from filename
        resource_name = self._create_resource_name(audio_path.name)

        # Setup directories
        override = Path(output_dir) if output_dir else None
        output_dir, logs_dir = self._setup_resource_directories(resource_name, output_dir_override=override)
        
        # Initialize error handler for this specific resource
        error_handler = ErrorHandler(f"audio_{resource_name}")
        # Move log file to resource-specific directory
        error_handler.log_file = logs_dir / "extraction.log"
        error_handler.logger = error_handler._setup_logger()
        
        # Validate audio file
        if not self._validate_audio_file(audio_path, error_handler):
            return self._create_error_result(
                resource_name,
                "Audio file validation failed",
                output_dir,
                audio_path.name
            )
        
        # Get file size
        file_size = audio_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        error_handler.log_info(
            f"Starting audio extraction: {audio_path.name}",
            metadata={
                "size_mb": f"{file_size_mb:.2f}",
                "resource_name": resource_name,
                "output_dir": str(output_dir)
            }
        )
        
        try:
            # Override language if provided
            if language:
                original_lang = WHISPER_LANGUAGE
                import utils.configs as config
                config.WHISPER_LANGUAGE = language
            
            # Transcribe audio
            transcription = self._transcribe_with_whisper(audio_path, error_handler)
            
            # Restore original language setting
            if language:
                import utils.configs as config
                config.WHISPER_LANGUAGE = original_lang
            
            if not transcription:
                raise ValueError("Transcription failed - no text returned")
            
            # Clean text if requested
            if clean_text:
                transcription = self.text_cleaner.clean_text(
                    transcription,
                    remove_urls=False,
                    remove_emails=False,
                    fix_spacing=True
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                "resource_name": resource_name,
                "resource_id": resource_id or resource_name,
                "filename": audio_path.name,
                "source_type": "audio",
                "audio_format": audio_path.suffix.lower(),
                "upload_date": datetime.now().isoformat(),
                "extraction_timestamp": datetime.now().isoformat(),
                "file_size_bytes": file_size,
                "processing_time_seconds": round(processing_time, 2),
                "status": "success",
                "error_message": None,
                "character_count": len(transcription),
                "word_count": len(transcription.split()),
                "whisper_model": WHISPER_MODEL,
                "language": language or "auto-detect"
            }
            
            # Save text file
            text_file = output_dir / f"{resource_name}_text.txt"
            text_file.write_text(transcription, encoding='utf-8')
            metadata["extracted_text_path"] = str(text_file)
            
            # Save metadata file
            metadata_file = output_dir / f"{resource_name}_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            error_handler.log_success(
                f"Audio extracted successfully: {audio_path.name}",
                metadata={
                    "chars": len(transcription),
                    "words": len(transcription.split()),
                    "time": f"{processing_time:.2f}s",
                    "output": str(output_dir)
                }
            )
            
            return {
                "success": True,
                "resource_name": resource_name,
                "resource_id": resource_id or resource_name,
                "text_file": str(text_file),
                "metadata_file": str(metadata_file),
                "output_dir": str(output_dir),
                "logs_dir": str(logs_dir),
                "metadata": metadata,
                "extracted_text": transcription
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            error_handler.log_error(
                e,
                context=f"Extracting audio: {audio_path.name}",
                metadata={"resource_name": resource_name}
            )
            
            return self._create_error_result(
                resource_name,
                str(e),
                output_dir,
                audio_path.name,
                file_size,
                processing_time
            )
    
    def _create_error_result(self,
                           resource_name: str,
                           error_message: str,
                           output_dir: Path,
                           filename: str = "unknown",
                           file_size: int = 0,
                           processing_time: float = 0) -> Dict[str, Any]:
        """Create error result when extraction fails"""
        
        metadata = {
            "resource_name": resource_name,
            "filename": filename,
            "source_type": "audio",
            "upload_date": datetime.now().isoformat(),
            "extraction_timestamp": datetime.now().isoformat(),
            "file_size_bytes": file_size,
            "processing_time_seconds": round(processing_time, 2),
            "status": "failed",
            "error_message": error_message
        }
        
        # Save metadata even for failed extraction
        metadata_file = output_dir / f"{resource_name}_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        return {
            "success": False,
            "resource_name": resource_name,
            "text_file": None,
            "metadata_file": str(metadata_file),
            "output_dir": str(output_dir),
            "metadata": metadata,
            "extracted_text": "",
            "error": error_message
        }


# Example usage and testing
if __name__ == "__main__":
    from utils.file_picker import FilePicker
    
    print("=== Testing Audio Extractor ===\n")
    
    # Check API key
    if GROQ_API_KEY_AUDIO == "your-groq-api-key-here":
        print("❌ ERROR: Groq API key not set!")
        print("   Please set your API key in one of these ways:")
        print("   1. Create a .env file with: GROQ_API_KEY=your_key_here")
        print("   2. Edit config.py and set GROQ_API_KEY")
        print("\n   Get your free API key at: https://console.groq.com/")
        exit(1)
    
    try:
        # Initialize extractor
        extractor = AudioExtractor()
        
        # Use file picker to select audio
        picker = FilePicker()
        print("Please select an audio file...")
        test_audio = picker.pick_audio()
        picker.close()
        
        if test_audio:
            print(f"\n✓ Selected: {Path(test_audio).name}\n")
            
            print("Starting transcription...")
            print("(This may take a moment depending on audio length)\n")
            
            result = extractor.extract(
                audio_path=test_audio,
                clean_text=True
            )
            
            if result['success']:
                print(f"   ✓ Success!")
                print(f"   Resource name: {result['resource_name']}")
                print(f"   Output directory: {result['output_dir']}")
                print(f"   Logs directory: {result['logs_dir']}")
                print(f"   Text file: {result['text_file']}")
                print(f"   Metadata file: {result['metadata_file']}")
                print(f"   Characters: {result['metadata']['character_count']}")
                print(f"   Words: {result['metadata']['word_count']}")
                print(f"   Processing time: {result['metadata']['processing_time_seconds']}s")
                print(f"   Model: {result['metadata']['whisper_model']}")
                
                print(f"\n   Transcription preview (first 400 chars):")
                print("   " + "-" * 50)
                preview = result['extracted_text'][:400]
                print(f"   {preview}...")
                print("   " + "-" * 50)
            else:
                print(f"   ✗ Failed: {result['error']}")
        else:
            print("❌ No file selected")
            print("   The extractor is ready to use!")
            
    except ValueError as e:
        print(f"❌ ERROR: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")