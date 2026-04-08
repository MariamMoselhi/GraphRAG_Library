"""
File Picker Utility for HoloLearn Extractor
Provides GUI file selection dialogs for testing.
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Optional, List


class FilePicker:
    """Simple file picker using tkinter"""
    
    def __init__(self):
        # Create root window but hide it
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        self.root.attributes('-topmost', True)  # Bring dialog to front
    
    def pick_file(self, 
                  title: str = "Select a file",
                  filetypes: Optional[List[tuple]] = None,
                  initial_dir: Optional[str] = None) -> Optional[str]:
        """
        Open file picker dialog and return selected file path
        
        Args:
            title: Dialog window title
            filetypes: List of (description, extension) tuples
                      Example: [("PDF files", "*.pdf"), ("All files", "*.*")]
            initial_dir: Starting directory for file browser
        
        Returns:
            Selected file path as string, or None if cancelled
        
        Example:
            picker = FilePicker()
            pdf_path = picker.pick_file(
                title="Select PDF",
                filetypes=[("PDF files", "*.pdf")]
            )
        """
        if filetypes is None:
            filetypes = [("All files", "*.*")]
        
        if initial_dir is None:
            initial_dir = str(Path.home())
        
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
            initialdir=initial_dir
        )
        
        return file_path if file_path else None
    
    def pick_multiple_files(self,
                           title: str = "Select files",
                           filetypes: Optional[List[tuple]] = None,
                           initial_dir: Optional[str] = None) -> List[str]:
        """
        Open file picker for multiple files
        
        Returns:
            List of selected file paths
        """
        if filetypes is None:
            filetypes = [("All files", "*.*")]
        
        if initial_dir is None:
            initial_dir = str(Path.home())
        
        file_paths = filedialog.askopenfilenames(
            title=title,
            filetypes=filetypes,
            initialdir=initial_dir
        )
        
        return list(file_paths) if file_paths else []
    
    def pick_pdf(self) -> Optional[str]:
        """Quick helper for PDF selection"""
        return self.pick_file(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
    
    def pick_pptx(self) -> Optional[str]:
        """Quick helper for PowerPoint selection"""
        return self.pick_file(
            title="Select PowerPoint file",
            filetypes=[
                ("PowerPoint files", "*.pptx *.ppt"),
                ("All files", "*.*")
            ]
        )
    
    def pick_docx(self) -> Optional[str]:
        """Quick helper for Word document selection"""
        return self.pick_file(
            title="Select Word document",
            filetypes=[
                ("Word documents", "*.docx *.doc"),
                ("All files", "*.*")
            ]
        )
    
    def pick_video(self) -> Optional[str]:
        """Quick helper for video selection"""
        return self.pick_file(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
    
    def pick_audio(self) -> Optional[str]:
        """Quick helper for audio selection"""
        return self.pick_file(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
                ("All files", "*.*")
            ]
        )
    
    def pick_image(self) -> Optional[str]:
        """Quick helper for image selection"""
        return self.pick_file(
            title="Select image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
    
    def pick_images(self) -> List[str]:
        """Quick helper for multiple image selection"""
        return self.pick_multiple_files(
            title="Select image files",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
    
    def close(self):
        """Clean up the tkinter root window"""
        try:
            self.root.destroy()
        except:
            pass


# Example usage
if __name__ == "__main__":
    print("=== Testing File Picker ===\n")
    
    picker = FilePicker()
    
    print("1. Opening PDF file picker...")
    pdf_path = picker.pick_pdf()
    
    if pdf_path:
        print(f"   ✓ Selected: {pdf_path}\n")
        
        print("2. Opening multiple image picker...")
        image_paths = picker.pick_images()
        
        if image_paths:
            print(f"   ✓ Selected {len(image_paths)} images:")
            for img in image_paths:
                print(f"      - {Path(img).name}")
        else:
            print("   ✗ No images selected")
    else:
        print("   ✗ No file selected")
    
    picker.close()
    print("\n✓ File picker test complete!")