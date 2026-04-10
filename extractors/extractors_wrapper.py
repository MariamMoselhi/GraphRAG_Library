from pathlib import Path

class SimpleExtractorWrapper:
    """
    Simple wrapper for extractors.
    Returns ONLY extracted text as a string.
    """

    def __init__(self):
        print("Extractor wrapper initialized")

    def extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        try:
            from extractors.pdf_extractor import PDFExtractor
            extractor = PDFExtractor()
            result = extractor.extract(file_path)
            if result["success"]:
                return result.get("extracted_text", "")
            
            print(f"PDF extraction failed: {result.get('error', 'Unknown error')}")
            return ""
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return ""

    def extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            from extractors.docx_extractor import DOCXExtractor
            extractor = DOCXExtractor()
            result = extractor.extract(file_path)
            if result["success"]:
                return result.get("extracted_text", "")
            print(f"DOCX extraction failed: {result.get('error', 'Unknown error')}")
            return ""
        except Exception as e:
            print(f"DOCX extraction failed: {e}")
            return ""

    def extract_pptx(self, file_path: str) -> str:
        """Extract text from PowerPoint"""
        try:
            from extractors.pptx_extractor import PPTXExtractor
            extractor = PPTXExtractor()
            result = extractor.extract(file_path)
            if result["success"]:
                return result.get("extracted_text", "")
            print(f"PPTX extraction failed: {result.get('error', 'Unknown error')}")
            return ""
        except Exception as e:
            print(f"PPTX extraction failed: {e}")
            return ""

    def extract_audio(self, file_path: str) -> str:
        """Extract transcript from audio"""
        try:
            from extractors.audio_extractor import AudioExtractor
            extractor = AudioExtractor()
            result = extractor.extract(file_path)
            if result["success"]:
                return result.get("extracted_text", "")
            print(f"Audio extraction failed: {result.get('error', 'Unknown error')}")
            return ""
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return ""

    def extract_video(self, file_path: str) -> str:
        """Extract transcript from video (audio + OCR)"""
        try:
            from extractors.video_extractor import VideoExtractor
            extractor = VideoExtractor()
            result = extractor.extract(file_path)
            if result["success"]:
                return result.get("clean_transcript", "") or result.get("extracted_text", "")
            print(f"Video extraction failed: {result.get('error', 'Unknown error')}")
            return ""
        except Exception as e:
            print(f"Video extraction failed: {e}")
            return ""

    def extract_url(self, url: str) -> str:
        """Extract text from URL"""
        try:
            from extractors.url_extractor import URLExtractor
            extractor = URLExtractor()
            result = extractor.extract(url)
            if result["success"]:
                return result.get("extracted_text", "")
            print(f"URL extraction failed: {result.get('error', 'Unknown error')}")
            return ""
        except Exception as e:
            print(f"URL extraction failed: {e}")
            return ""

    def extract_text(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Text file extraction failed: {e}")
            return ""

    def extract_auto(self, file_path: str) -> str:
        """Auto-detect file type and extract text"""
        ext = Path(file_path).suffix.lower()

        extractors = {
            ".pdf": self.extract_pdf,
            ".docx": self.extract_docx,
            ".doc": self.extract_docx,
            ".pptx": self.extract_pptx,
            ".ppt": self.extract_pptx,
            ".txt": self.extract_text,
            ".md": self.extract_text,
            ".mp3": self.extract_audio,
            ".wav": self.extract_audio,
            ".m4a": self.extract_audio,
            ".flac": self.extract_audio,
            ".ogg": self.extract_audio,
            ".aac": self.extract_audio,
            ".mp4": self.extract_video,
            ".avi": self.extract_video,
            ".mov": self.extract_video,
            ".mkv": self.extract_video,
            ".flv": self.extract_video,
            ".wmv": self.extract_video,
        }

        handler = extractors.get(ext)
        if handler:
            return handler(file_path)

        # Try as URL if no extension match
        if file_path.startswith(("http://", "https://", "www.")):
            return self.extract_url(file_path)

        print(f"Unsupported file type: {ext}")
        return ""
