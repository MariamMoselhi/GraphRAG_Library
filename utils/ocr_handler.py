import os
from pathlib import Path
from typing import List, Dict, Optional, Union

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Import our utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.configs import (
    EASYOCR_GPU, PIX2TEX_DEVICE, EASYOCR_LANGUAGES, _AUTO_CONFIG,
    ENABLE_SMART_MATH_DETECTION, MATH_SYMBOLS
)
from utils.error_handler import ErrorHandler
from utils.text_cleaner import TextCleaner

# Note: GPU is always attempted first in OCRHandler.__init__
# These are only used as reference, not to control EasyOCR GPU
DEVICE = _AUTO_CONFIG.get('device', 'cpu')
USE_GPU = EASYOCR_GPU


class OCRHandler:
    """Handle OCR operations with EasyOCR and pix2tex (LaTeX-OCR)"""

    def __init__(self, languages: List[str] = None):
        """
        Initialize OCR handler

        Args:
            languages: List of language codes for OCR (default: from config)
                      Examples: ['en'], ['en', 'ar'], ['en', 'fr', 'es']
        """
        if not _EASYOCR_AVAILABLE:
            raise ImportError(
                "easyocr not installed. Install with: pip install easyocr"
            )
        if not _PIL_AVAILABLE:
            raise ImportError(
                "Pillow not installed. Install with: pip install Pillow"
            )

        self.error_handler = ErrorHandler("ocr_handler")
        self.languages = languages or EASYOCR_LANGUAGES
        self.text_cleaner = TextCleaner()

        # Initialize EasyOCR reader - try GPU first, fall back to CPU
        self.use_gpu = False

        # Always attempt GPU first, let EasyOCR handle its own CUDA detection
        try:
            self.error_handler.log_info(
                "Initializing EasyOCR with GPU...",
                metadata={"languages": self.languages}
            )
            self.reader = easyocr.Reader(
                self.languages,
                gpu=True,
                verbose=False
            )
            self.use_gpu = True
            self.error_handler.log_success(
                "EasyOCR initialized on GPU",
                metadata={"languages": self.languages}
            )
        except Exception as gpu_err:
            # GPU failed - fall back to CPU
            self.error_handler.log_warning(
                f"GPU init failed ({gpu_err}), falling back to CPU",
                metadata={"languages": self.languages}
            )
            try:
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=False,
                    verbose=False
                )
                self.error_handler.log_success(
                    "EasyOCR initialized on CPU (fallback)",
                    metadata={"languages": self.languages}
                )
            except Exception as cpu_err:
                self.error_handler.log_error(
                    cpu_err,
                    context="Initializing EasyOCR",
                    metadata={"languages": self.languages}
                )
                raise

        # Try to initialize pix2tex (LaTeX-OCR) for math equations
        self.latex_ocr_available = False
        self.latex_ocr = None
        try:
            from pix2tex.cli import LatexOCR
            self.latex_ocr = LatexOCR()
            self.latex_ocr_available = True
            self.error_handler.log_success("pix2tex LatexOCR initialized for equation recognition")
        except ImportError:
            self.error_handler.log_warning(
                "pix2tex not available - equations will not be recognized",
                metadata={"solution": "Install with: pip install pix2tex"}
            )
        except Exception as e:
            self.error_handler.log_warning(
                "pix2tex LatexOCR initialization failed - equations will not be recognized",
                metadata={"error": str(e)}
            )
    
    def extract_text_from_image(self, 
                               image_path: str,
                               detect_equations: bool = True,
                               confidence_threshold: float = 0.5) -> Dict[str, str]:
        """
        Extract text from a single image
        
        Args:
            image_path: Path to image file
            detect_equations: Whether to detect math equations
            confidence_threshold: Minimum confidence for text detection (0.0-1.0)
        
        Returns:
            Dictionary with 'text', 'equations', and 'combined'
        
        Example:
            ocr = OCRHandler()
            result = ocr.extract_text_from_image("slide.jpg")
            print(result['combined'])  # All text + equations
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            self.error_handler.log_info(
                f"Processing image: {image_path.name}",
                metadata={"detect_equations": detect_equations}
            )
            
            result = {
                'text': '',
                'equations': '',
                'combined': ''
            }
            
            # Extract general text with EasyOCR
            text = self._extract_with_easyocr(
                str(image_path),
                confidence_threshold
            )
            result['text'] = text
            
            # Extract equations with pix2tex (LaTeX-OCR) if available
            # Smart detection: only run the heavier model if math content is likely
            if detect_equations and self.latex_ocr_available:
                should_run = True
                if ENABLE_SMART_MATH_DETECTION and result['text']:
                    should_run = self._has_math_content(result['text'])
                if should_run:
                    equations = self._extract_equations(str(image_path))
                    result['equations'] = equations
            
            # Combine results
            combined_parts = []
            if result['text']:
                combined_parts.append(result['text'])
            if result['equations']:
                combined_parts.append(f"\n[EQUATIONS]\n{result['equations']}")
            
            result['combined'] = '\n'.join(combined_parts)
            
            self.error_handler.log_success(
                f"Extracted text from {image_path.name}",
                metadata={
                    "text_length": len(result['text']),
                    "has_equations": bool(result['equations'])
                }
            )
            
            return result
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                context=f"Extracting text from {image_path}",
                metadata={"image": str(image_path)}
            )
            # Return empty result instead of crashing
            return {'text': '', 'equations': '', 'combined': ''}
    
    def extract_text_from_images(self,
                                image_paths: List[str],
                                detect_equations: bool = True,
                                confidence_threshold: float = 0.5) -> Dict[str, str]:
        """
        Extract text from multiple images and combine
        
        Args:
            image_paths: List of paths to image files
            detect_equations: Whether to detect math equations
            confidence_threshold: Minimum confidence for text detection
        
        Returns:
            Dictionary with 'combined', 'text_only', 'equations_only'
        
        Example:
            ocr = OCRHandler()
            frames = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
            result = ocr.extract_text_from_images(frames)
            print(result['text_only'])       # Just text
            print(result['equations_only'])  # Just equations
            print(result['combined'])        # Both together
        """
        self.error_handler.log_info(
            f"Processing {len(image_paths)} images...",
            metadata={"count": len(image_paths)}
        )
        
        all_text = []
        all_equations = []
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.extract_text_from_image(
                    image_path,
                    detect_equations,
                    confidence_threshold
                )
                
                if result['text'] or result['equations']:
                    # Add frame marker for text
                    if result['text']:
                        all_text.append(f"--- Frame {i} ---")
                        all_text.append(result['text'])
                        all_text.append("")  # Blank line
                    
                    # Add frame marker for equations
                    if result['equations']:
                        all_equations.append(f"--- Frame {i} ---")
                        all_equations.append(result['equations'])
                        all_equations.append("")  # Blank line
                    
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                self.error_handler.log_warning(
                    f"Failed to process image {i}/{len(image_paths)}",
                    metadata={"image": image_path, "error": str(e)}
                )
        
        text_only = '\n'.join(all_text)
        equations_only = '\n'.join(all_equations)
        
        # Combined version
        combined_parts = []
        if text_only:
            combined_parts.append(text_only)
        if equations_only:
            combined_parts.append(f"\n[EQUATIONS]\n{equations_only}")
        combined = '\n'.join(combined_parts)
        
        # Clean the text
        text_only = self.text_cleaner.clean_text(text_only)
        combined = self.text_cleaner.clean_text(combined)
        
        self.error_handler.log_success(
            f"Processed {successful}/{len(image_paths)} images successfully",
            metadata={"successful": successful, "failed": failed}
        )
        
        return {
            'combined': combined,
            'text_only': text_only,
            'equations_only': equations_only
        }
    
    def _extract_with_easyocr(self, 
                             image_path: str,
                             confidence_threshold: float = 0.5) -> str:
        """
        Extract text using EasyOCR
        
        Returns:
            Extracted text as string
        """
        try:
            # Read text from image
            # Returns list of ([bbox], text, confidence)
            results = self.reader.readtext(image_path)
            
            # Filter by confidence and extract text
            texts = []
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    texts.append(text)
            
            # Join with spaces
            extracted_text = ' '.join(texts)
            
            # Clean the text
            extracted_text = self.text_cleaner.clean_text(extracted_text)
            
            return extracted_text
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                context=f"EasyOCR processing",
                metadata={"image": image_path}
            )
            return ""
    
    def _has_math_content(self, text: str) -> bool:
        """
        Check if text likely contains mathematical content.

        Uses MATH_SYMBOLS from config plus keyword patterns to decide
        whether it's worth running the heavier LaTeX-OCR model.
        """
        if not text:
            return False

        # Check for Unicode math symbols from config
        for sym in MATH_SYMBOLS:
            if sym in text:
                return True

        # Check for keyword patterns that suggest math/equations
        text_lower = text.lower()
        math_keywords = [
            'equation', 'formula', 'theorem', 'proof', 'lemma',
            'derivative', 'integral', 'matrix', 'vector',
            'sin', 'cos', 'tan', 'log', 'ln', 'lim',
            'f(x)', 'g(x)', 'dx', 'dy',
        ]
        for kw in math_keywords:
            if kw in text_lower:
                return True

        # Check for patterns like "x^2", "a_1", "2x + 3", numeric expressions with operators
        import re
        math_patterns = [
            r'[a-zA-Z]\s*[\^_]\s*\d',        # x^2, a_1
            r'\d+\s*[+\-*/]\s*\d+',           # 2 + 3, 4*5
            r'[a-zA-Z]\s*\(\s*[a-zA-Z]',      # f(x), g(t)
            r'\d+\s*[a-zA-Z]\s*[+\-=]',       # 2x +, 3y =
            r'[a-zA-Z]\s*=\s*[a-zA-Z0-9]',    # y = 2, x = a
        ]
        for pat in math_patterns:
            if re.search(pat, text):
                return True

        return False

    def _extract_equations(self, image_path: str) -> str:
        """
        Extract mathematical equations using pix2tex (LaTeX-OCR).

        pix2tex takes a PIL Image and returns a LaTeX string.

        Returns:
            LaTeX equation string, or "" if extraction fails/unavailable.
        """
        if not self.latex_ocr_available or self.latex_ocr is None:
            return ""

        try:
            img = Image.open(image_path)
            # pix2tex LatexOCR: calling the model on a PIL image returns LaTeX
            latex = self.latex_ocr(img)

            if latex and isinstance(latex, str):
                latex = latex.strip()
                # Skip trivially empty or whitespace-only results
                if len(latex) < 2:
                    return ""
                return latex

            return ""

        except Exception as e:
            self.error_handler.log_warning(
                "pix2tex equation extraction failed",
                metadata={"image": image_path, "error": str(e)}
            )
            return ""
    
    def is_image_readable(self, image_path: str) -> bool:
        """
        Check if image can be opened and is valid
        
        Returns:
            True if image is readable, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
            return True
        except Exception:
            return False


# Example usage
if __name__ == "__main__":
    print("=== Testing OCR Handler ===\n")
    
    # Initialize OCR
    print("1. Initializing OCR Handler...")
    ocr = OCRHandler()
    print(f"   Device: {DEVICE}")
    print(f"   GPU enabled: {USE_GPU}")
    print(f"   LaTeX-OCR available: {ocr.latex_ocr_available}\n")
    
    # Test with sample image (you'll need to provide one)
    test_image = "test_image.jpg"  # Replace with actual image path
    
    if Path(test_image).exists():
        print("2. Testing single image extraction...")
        result = ocr.extract_text_from_image(test_image)
        print(f"   Text extracted: {len(result['text'])} characters")
        print(f"   Equations found: {bool(result['equations'])}")
        print(f"\n   Preview:")
        print("   " + "-" * 50)
        preview = result['combined'][:200] + "..." if len(result['combined']) > 200 else result['combined']
        print(f"   {preview}")
        print("   " + "-" * 50)
        
    else:
        print(f"âŒ Test image not found: {test_image}")
        print("   Place a test image (screenshot with text) to run tests")
    
    # Test batch processing
    test_images = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
    existing_images = [img for img in test_images if Path(img).exists()]
    
    if existing_images:
        print(f"\n3. Testing batch extraction ({len(existing_images)} images)...")
        result = ocr.extract_text_from_images(existing_images)
        print(f"   Total text extracted: {len(result['combined'])} characters")
        print(f"   Text only: {len(result['text_only'])} characters")
        print(f"   Equations only: {len(result['equations_only'])} characters")
        print(f"\n   Preview:")
        print("   " + "-" * 50)
        preview = result['combined'][:300] + "..." if len(result['combined']) > 300 else result['combined']
        print(f"   {preview}")
        print("   " + "-" * 50)