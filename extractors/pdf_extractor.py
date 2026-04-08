"""
PDF Extractor for HoloLearn
Extracts text content from PDF files.
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
from utils.configs import OUTPUT_DIR, LOGS_DIR, MAX_PDF_SIZE
from utils.error_handler import ErrorHandler
from utils.text_cleaner import TextCleaner

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


class PDFExtractor:
    """Extract text from PDF files"""
    
    def __init__(self):
        if not _FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF not installed. Install with: pip install PyMuPDF"
            )

        self.text_cleaner = TextCleaner()
        self.base_output_dir = OUTPUT_DIR
        self.base_logs_dir = LOGS_DIR

        # Ensure base directories exist
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_logs_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_resource_name(self, filename: str) -> str:
        """
        Create a clean resource name from filename
        
        Example: "Lecture 5 - Machine Learning.pdf" → "lecture_5_machine_learning"
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
        
        return name or "unnamed_resource"
    
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
    
    def extract(self,
                pdf_path: str,
                resource_id: Optional[str] = None,
                clean_text: bool = True,
                password: Optional[str] = None,
                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from a PDF file

        Args:
            pdf_path: Path to PDF file
            resource_id: Optional unique identifier (if None, uses filename)
            clean_text: Whether to clean the extracted text
            password: Password if PDF is encrypted (optional)
            output_dir: Optional shared output directory. When provided,
                all output files are written here instead of a per-resource subfolder.

        Returns:
            Dictionary with extraction results and metadata

        Example:
            extractor = PDFExtractor()
            result = extractor.extract("lecture.pdf")
            # Creates: output/lecture/text.txt and output/lecture/metadata.json
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        # Create resource name from filename
        resource_name = self._create_resource_name(pdf_path.name)

        # Setup directories
        override = Path(output_dir) if output_dir else None
        output_dir, logs_dir = self._setup_resource_directories(resource_name, output_dir_override=override)
        
        # Initialize error handler for this specific resource
        error_handler = ErrorHandler(f"pdf_{resource_name}")
        # Move log file to resource-specific directory
        error_handler.log_file = logs_dir / "extraction.log"
        error_handler.logger = error_handler._setup_logger()
        
        # Validate file
        if not pdf_path.exists():
            error_msg = f"PDF file not found: {pdf_path}"
            error_handler.log_error(
                FileNotFoundError(error_msg),
                context="Validating PDF file",
                metadata={"path": str(pdf_path)}
            )
            return self._create_error_result(
                resource_name, error_msg, output_dir, pdf_path.name
            )
        
        # Check file size
        file_size = pdf_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > MAX_PDF_SIZE:
            error_msg = f"PDF too large: {file_size_mb:.2f}MB (max: {MAX_PDF_SIZE}MB)"
            error_handler.log_error(
                ValueError(error_msg),
                context="Checking PDF size",
                metadata={"size_mb": file_size_mb, "max_mb": MAX_PDF_SIZE}
            )
            return self._create_error_result(
                resource_name, error_msg, output_dir, pdf_path.name, file_size
            )
        
        error_handler.log_info(
            f"Starting PDF extraction: {pdf_path.name}",
            metadata={
                "size_mb": f"{file_size_mb:.2f}",
                "resource_name": resource_name,
                "output_dir": str(output_dir)
            }
        )
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Check if encrypted
            if doc.is_encrypted:
                if password:
                    if not doc.authenticate(password):
                        raise ValueError("Invalid password for encrypted PDF")
                else:
                    raise ValueError("PDF is encrypted but no password provided")
            
            # Extract text from all pages
            extracted_text = ""
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():  # Only add if page has text
                    extracted_text += f"\n--- Page {page_num + 1} ---\n"
                    extracted_text += page_text
                    extracted_text += "\n"
            
            doc.close()
            
            # Clean text if requested
            if clean_text:
                extracted_text = self.text_cleaner.clean_text(
                    extracted_text,
                    remove_urls=False,  # Keep URLs in educational content
                    remove_emails=False,  # Keep emails
                    fix_spacing=True
                )
            
            # Remove duplicate lines (PDFs often have duplicates)
            extracted_text = self.text_cleaner.remove_duplicate_lines(extracted_text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                "resource_name": resource_name,
                "resource_id": resource_id or resource_name,
                "filename": pdf_path.name,
                "source_type": "pdf",
                "upload_date": datetime.now().isoformat(),
                "extraction_timestamp": datetime.now().isoformat(),
                "file_size_bytes": file_size,
                "processing_time_seconds": round(processing_time, 2),
                "status": "success",
                "error_message": None,
                "page_count": page_count,
                "character_count": len(extracted_text),
                "is_encrypted": False
            }
            
            # Save text file
            text_file = output_dir / f"{resource_name}_text.txt"
            text_file.write_text(extracted_text, encoding='utf-8')
            metadata["extracted_text_path"] = str(text_file)

            # Save metadata file
            metadata_file = output_dir / f"{resource_name}_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

            error_handler.log_success(
                f"PDF extracted successfully: {pdf_path.name}",
                metadata={
                    "pages": page_count,
                    "chars": len(extracted_text),
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
                "extracted_text": extracted_text
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            error_handler.log_error(
                e,
                context=f"Extracting PDF: {pdf_path.name}",
                metadata={"resource_name": resource_name}
            )
            
            return self._create_error_result(
                resource_name,
                str(e),
                output_dir,
                pdf_path.name,
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
            "source_type": "pdf",
            "upload_date": datetime.now().isoformat(),
            "extraction_timestamp": datetime.now().isoformat(),
            "file_size_bytes": file_size,
            "processing_time_seconds": round(processing_time, 2),
            "status": "failed",
            "error_message": error_message
        }
        
        # Save metadata even for failed extraction
        metadata_file = output_dir / "metadata.json"
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
    
    def extract_metadata_only(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract only metadata without extracting text
        Useful for quick file info
        
        Returns:
            Dictionary with PDF metadata
        """
        try:
            pdf_path = Path(pdf_path)
            doc = fitz.open(pdf_path)
            
            metadata = {
                "filename": pdf_path.name,
                "resource_name": self._create_resource_name(pdf_path.name),
                "page_count": len(doc),
                "is_encrypted": doc.is_encrypted,
                "file_size_bytes": pdf_path.stat().st_size,
                "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
                "pdf_metadata": doc.metadata  # Title, author, etc.
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            error_handler = ErrorHandler("pdf_metadata")
            error_handler.log_error(
                e,
                context=f"Extracting metadata from {pdf_path}"
            )
            return {}


# Example usage and testing
if __name__ == "__main__":
    from utils.file_picker import FilePicker
    
    print("=== Testing PDF Extractor ===\n")
    
    # Initialize extractor
    extractor = PDFExtractor()
    
    # Use file picker to select PDF
    picker = FilePicker()
    print("Please select a PDF file...")
    test_pdf = picker.pick_pdf()
    picker.close()
    
    if test_pdf:
        print(f"\n✓ Selected: {Path(test_pdf).name}\n")
        
        print("1. Extracting metadata only...")
        metadata = extractor.extract_metadata_only(test_pdf)
        print(f"   Resource name: {metadata.get('resource_name', 'N/A')}")
        print(f"   Pages: {metadata.get('page_count', 'N/A')}")
        print(f"   Size: {metadata.get('file_size_mb', 'N/A')} MB")
        print(f"   Encrypted: {metadata.get('is_encrypted', 'N/A')}\n")
        
        print("2. Full extraction...")
        result = extractor.extract(
            pdf_path=test_pdf,
            clean_text=True
        )
        
        if result['success']:
            print(f"   ✓ Success!")
            print(f"   Resource name: {result['resource_name']}")
            print(f"   Output directory: {result['output_dir']}")
            print(f"   Logs directory: {result['logs_dir']}")
            print(f"   Text file: {result['text_file']}")
            print(f"   Metadata file: {result['metadata_file']}")
            print(f"   Pages: {result['metadata']['page_count']}")
            print(f"   Characters: {result['metadata']['character_count']}")
            print(f"   Processing time: {result['metadata']['processing_time_seconds']}s")
            
            print(f"\n   Preview (first 300 chars):")
            print("   " + "-" * 50)
            preview = result['extracted_text'][:300]
            print(f"   {preview}...")
            print("   " + "-" * 50)
        else:
            print(f"   ✗ Failed: {result['error']}")
    else:
        print("❌ No file selected")
        print("   The extractor is ready to use!")