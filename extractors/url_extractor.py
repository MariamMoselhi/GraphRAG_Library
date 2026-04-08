"""
URL Extractor for HoloLearn
Extracts text content from web pages.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import time
import re
from urllib.parse import urlparse

# Import our utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.configs import (
    OUTPUT_DIR, 
    LOGS_DIR, 
    REQUEST_TIMEOUT,
    USER_AGENT,
    MAX_RETRIES,
    RETRY_DELAY,
    BACKOFF_FACTOR
)
from utils.error_handler import ErrorHandler
from utils.text_cleaner import TextCleaner

try:
    import requests
    from bs4 import BeautifulSoup
    _WEB_DEPS_AVAILABLE = True
except ImportError as _web_import_err:
    _WEB_DEPS_AVAILABLE = False
    _web_import_err_msg = str(_web_import_err)


class URLExtractor:
    """Extract text from web pages"""
    
    def __init__(self):
        if not _WEB_DEPS_AVAILABLE:
            raise ImportError(
                f"Web dependencies missing: {_web_import_err_msg}. "
                "Install with: pip install requests beautifulsoup4"
            )

        self.text_cleaner = TextCleaner()
        self.base_output_dir = OUTPUT_DIR
        self.base_logs_dir = LOGS_DIR

        # Ensure base directories exist
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup requests session with retries
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
    
    def _create_resource_name(self, url: str) -> str:
        """
        Create a clean resource name from URL
        
        Example: "https://example.com/ml-basics" → "example_com_ml_basics"
        """
        # Parse URL
        parsed = urlparse(url)
        
        # Get domain and path
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.strip('/')
        
        # Combine domain and path
        if path:
            name = f"{domain}_{path}"
        else:
            name = domain
        
        # Convert to lowercase
        name = name.lower()
        
        # Replace special chars with underscore
        name = re.sub(r'[^\w\s-]', '_', name)
        name = re.sub(r'[-\s]+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Limit length
        if len(name) > 50:
            name = name[:50]
        
        return name or "webpage"
    
    def _setup_resource_directories(self, resource_name: str, output_dir_override: Optional[Path] = None) -> tuple:
        """
        Create directories for a specific resource

        Args:
            resource_name: Clean name derived from the input URL.
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
    
    def _fetch_url(self, url: str, error_handler: ErrorHandler) -> Optional[str]:
        """
        Fetch HTML content from URL with retries
        
        Returns:
            HTML content as string, or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                error_handler.log_info(
                    f"Fetching URL (attempt {attempt + 1}/{MAX_RETRIES})...",
                    metadata={"url": url}
                )
                
                response = self.session.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True
                )
                
                response.raise_for_status()  # Raise error for bad status codes
                
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type.lower():
                    error_handler.log_warning(
                        f"URL may not be HTML: {content_type}",
                        metadata={"url": url}
                    )
                
                return response.text
                
            except requests.exceptions.Timeout:
                error_handler.log_warning(
                    f"Timeout on attempt {attempt + 1}",
                    metadata={"url": url}
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (BACKOFF_FACTOR ** attempt))
                
            except requests.exceptions.RequestException as e:
                error_handler.log_warning(
                    f"Request failed on attempt {attempt + 1}: {str(e)}",
                    metadata={"url": url}
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (BACKOFF_FACTOR ** attempt))
        
        return None
    
    def _extract_text_from_html(self, html: str, url: str) -> str:
        """
        Extract main text content from HTML
        
        Returns:
            Extracted text
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Look for common content containers
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.post-content', '.article-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            main_content = soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Try to extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = f"Title: {title_tag.get_text().strip()}\n"
            title += f"URL: {url}\n"
            title += "=" * 60 + "\n\n"
        
        return title + text
    
    def extract(self,
                url: str,
                resource_id: Optional[str] = None,
                clean_text: bool = True,
                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from a web page

        Args:
            url: URL to scrape
            resource_id: Optional unique identifier (if None, uses URL)
            clean_text: Whether to clean the extracted text
            output_dir: Optional shared output directory. When provided,
                all output files are written here instead of a per-resource subfolder.

        Returns:
            Dictionary with extraction results and metadata

        Example:
            extractor = URLExtractor()
            result = extractor.extract("https://example.com/article")
            # Creates: output/example_com_article/example_com_article_text.txt
        """
        start_time = time.time()

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Create resource name from URL
        resource_name = self._create_resource_name(url)

        # Setup directories
        override = Path(output_dir) if output_dir else None
        output_dir, logs_dir = self._setup_resource_directories(resource_name, output_dir_override=override)
        
        # Initialize error handler for this specific resource
        error_handler = ErrorHandler(f"url_{resource_name}")
        # Move log file to resource-specific directory
        error_handler.log_file = logs_dir / "extraction.log"
        error_handler.logger = error_handler._setup_logger()
        
        error_handler.log_info(
            f"Starting URL extraction: {url}",
            metadata={
                "resource_name": resource_name,
                "output_dir": str(output_dir)
            }
        )
        
        try:
            # Fetch HTML content
            html = self._fetch_url(url, error_handler)
            
            if not html:
                raise ValueError("Failed to fetch URL after all retry attempts")
            
            # Extract text from HTML
            extracted_text = self._extract_text_from_html(html, url)
            
            if not extracted_text.strip():
                raise ValueError("No text content extracted from URL")
            
            # Clean text if requested
            if clean_text:
                extracted_text = self.text_cleaner.clean_text(
                    extracted_text,
                    remove_urls=False,  # Keep URLs for reference
                    remove_emails=False,  # Keep emails
                    fix_spacing=True
                )
            
            # Remove duplicate lines
            extracted_text = self.text_cleaner.remove_duplicate_lines(extracted_text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                "resource_name": resource_name,
                "resource_id": resource_id or resource_name,
                "url": url,
                "source_type": "url",
                "upload_date": datetime.now().isoformat(),
                "extraction_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(processing_time, 2),
                "status": "success",
                "error_message": None,
                "character_count": len(extracted_text),
                "word_count": len(extracted_text.split()),
                "domain": urlparse(url).netloc
            }
            
            # Save text file
            text_file = output_dir / f"{resource_name}_text.txt"
            text_file.write_text(extracted_text, encoding='utf-8')
            metadata["extracted_text_path"] = str(text_file)
            
            # Save metadata file
            metadata_file = output_dir / f"{resource_name}_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            error_handler.log_success(
                f"URL extracted successfully: {url}",
                metadata={
                    "chars": len(extracted_text),
                    "words": len(extracted_text.split()),
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
                context=f"Extracting URL: {url}",
                metadata={"resource_name": resource_name}
            )
            
            return self._create_error_result(
                resource_name,
                str(e),
                output_dir,
                url,
                processing_time
            )
    
    def _create_error_result(self,
                           resource_name: str,
                           error_message: str,
                           output_dir: Path,
                           url: str = "unknown",
                           processing_time: float = 0) -> Dict[str, Any]:
        """Create error result when extraction fails"""
        
        metadata = {
            "resource_name": resource_name,
            "url": url,
            "source_type": "url",
            "upload_date": datetime.now().isoformat(),
            "extraction_timestamp": datetime.now().isoformat(),
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
    
    def extract_multiple_urls(self, urls: list) -> Dict[str, Any]:
        """
        Extract text from multiple URLs
        
        Args:
            urls: List of URLs to scrape
        
        Returns:
            Dictionary with results for all URLs
        """
        results = {
            "total": len(urls),
            "successful": 0,
            "failed": 0,
            "extractions": []
        }
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Processing: {url}")
            
            result = self.extract(url)
            results["extractions"].append(result)
            
            if result["success"]:
                results["successful"] += 1
                print(f"✓ Success: {result['resource_name']}")
            else:
                results["failed"] += 1
                print(f"✗ Failed: {result['error']}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing URL Extractor ===\n")
    
    # Initialize extractor
    extractor = URLExtractor()
    
    # Option 1: Manual URL input
    print("Enter a URL to extract (or press Enter to skip):")
    user_url = input("> ").strip()
    
    # Option 2: Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.python.org/about/",
    ]
    
    if user_url:
        # Extract user-provided URL
        print(f"\nExtracting: {user_url}\n")
        result = extractor.extract(user_url, clean_text=True)
        
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
            
            print(f"\n   Preview (first 400 chars):")
            print("   " + "-" * 50)
            preview = result['extracted_text'][:400]
            print(f"   {preview}...")
            print("   " + "-" * 50)
        else:
            print(f"   ✗ Failed: {result['error']}")
    
    else:
        # Extract test URLs
        print(f"No URL provided. Testing with {len(test_urls)} example URLs...\n")
        results = extractor.extract_multiple_urls(test_urls)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total URLs: {results['total']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"{'='*60}")
# ```

# ---

# ## **Features:**

# ✅ **Web scraping** - Extracts main content from web pages  
# ✅ **Smart content detection** - Finds main article/content area  
# ✅ **Retry logic** - Automatic retries with exponential backoff  
# ✅ **HTML cleaning** - Removes scripts, styles, navigation  
# ✅ **Multiple URLs** - Batch processing support  
# ✅ **Resource naming from URL** - Clean folder names  
# ✅ **Error handling** - Network timeouts, bad URLs  

# ---

# ## **Output Example:**

# **Input:** `"https://en.wikipedia.org/wiki/Machine_learning"`

# **Creates:**
# ```
# output/en_wikipedia_org_wiki_machine_learning/
#   en_wikipedia_org_wiki_machine_learning_text.txt
#   en_wikipedia_org_wiki_machine_learning_metadata.json
  
# logs/en_wikipedia_org_wiki_machine_learning/
#   extraction.log
# ```

# **en_wikipedia_org_wiki_machine_learning_text.txt:**
# ```
# Title: Machine learning - Wikipedia
# URL: https://en.wikipedia.org/wiki/Machine_learning
# ============================================================

# Machine learning

# Machine learning (ML) is a field of study in artificial intelligence 
# concerned with the development and study of statistical algorithms that 
# can learn from data and generalize to unseen data...