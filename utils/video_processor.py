"""
Video Processor for HoloLearn Extractor
Handles video processing using FFmpeg: audio extraction and frame extraction.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Optional
from fractions import Fraction

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.configs import TEMP_DIR
from utils.error_handler import ErrorHandler


class VideoProcessor:
    """Process videos using FFmpeg"""

    def __init__(self):
        self.error_handler = ErrorHandler("video_processor")
        self.temp_dir = TEMP_DIR

        # Check if FFmpeg is installed
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is installed and accessible"""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.error_handler.log_error(
                Exception("FFmpeg not found"),
                context="Checking FFmpeg installation",
                metadata={"solution": "Install FFmpeg: https://ffmpeg.org/download.html"}
            )
            raise RuntimeError(
                "FFmpeg is not installed. Please install it from https://ffmpeg.org/download.html"
            )

    def extract_audio(self,
                      video_path: str,
                      output_path: Optional[str] = None) -> str:
        """
        Extract audio track from video and save as WAV file.

        Args:
            video_path: Path to input video file
            output_path: Where to save audio (if None, saves to temp dir)

        Returns:
            Path to extracted audio file
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_path is None:
            output_path = self.temp_dir / f"{video_path.stem}_audio.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.error_handler.log_info(
            f"Extracting audio from: {video_path.name}",
            metadata={"output": str(output_path)}
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path)
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(errors='replace') if e.stderr else str(e)
            self.error_handler.log_error(
                e,
                context=f"Extracting audio from {video_path.name}",
                metadata={"ffmpeg_error": error_msg}
            )
            raise

        self.error_handler.log_success(
            "Audio extracted successfully",
            metadata={
                "input": video_path.name,
                "output": output_path.name,
                "size_mb": round(output_path.stat().st_size / 1024 / 1024, 2)
            }
        )

        return str(output_path)

    def extract_frames(self,
                       video_path: str,
                       output_dir: Optional[str] = None,
                       fps: float = 0.5,
                       max_frames: int = 30) -> List[str]:
        """
        Extract frames from video at specified interval.

        Args:
            video_path: Path to input video
            output_dir: Where to save frames (if None, saves to temp dir)
            fps: Frames per second to extract (0.5 = 1 frame every 2 seconds)
            max_frames: Maximum number of frames to extract

        Returns:
            List of paths to extracted frame images
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_dir is None:
            output_dir = self.temp_dir / f"{video_path.stem}_frames"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        self.error_handler.log_info(
            f"Extracting frames from: {video_path.name}",
            metadata={"fps": fps, "max_frames": max_frames, "output_dir": str(output_dir)}
        )

        output_pattern = str(output_dir / f"{video_path.stem}_frame_%05d.png")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-vframes", str(max_frames),
            str(output_pattern)
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(errors='replace') if e.stderr else str(e)
            self.error_handler.log_error(
                e,
                context=f"Extracting frames from {video_path.name}",
                metadata={"ffmpeg_error": error_msg}
            )
            raise

        frame_files = sorted(output_dir.glob(f"{video_path.stem}_frame_*.png"))
        frame_paths = [str(f) for f in frame_files]

        self.error_handler.log_success(
            f"Extracted {len(frame_paths)} frames",
            metadata={"input": video_path.name, "frames": len(frame_paths)}
        )

        return frame_paths

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata (duration, resolution, codec, etc.)

        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            probe = json.loads(result.stdout.decode())

            video_stream = next(
                (s for s in probe.get('streams', []) if s.get('codec_type') == 'video'),
                None
            )

            if not video_stream:
                raise ValueError("No video stream found")

            # Safe FPS parsing using Fraction instead of eval()
            fps_str = video_stream.get('r_frame_rate', '0/1')
            try:
                fps_value = float(Fraction(fps_str))
            except (ValueError, ZeroDivisionError):
                fps_value = 0.0

            info = {
                "duration_seconds": float(probe.get('format', {}).get('duration', 0)),
                "width": int(video_stream.get('width', 0)),
                "height": int(video_stream.get('height', 0)),
                "codec": video_stream.get('codec_name', 'unknown'),
                "fps": fps_value,
                "size_mb": round(int(probe.get('format', {}).get('size', 0)) / 1024 / 1024, 2)
            }

            return info

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(errors='replace') if e.stderr else str(e)
            self.error_handler.log_error(
                e,
                context=f"Getting video info for {video_path}",
                metadata={"ffprobe_error": error_msg}
            )
            raise
        except Exception as e:
            self.error_handler.log_error(
                e,
                context=f"Getting video info for {video_path}"
            )
            raise


if __name__ == "__main__":
    processor = VideoProcessor()

    test_video = "test_video.mp4"

    if Path(test_video).exists():
        print("=== Testing Video Processor ===\n")

        print("1. Getting video info...")
        info = processor.get_video_info(test_video)
        print(f"   Duration: {info['duration_seconds']:.1f} seconds")
        print(f"   Resolution: {info['width']}x{info['height']}")
        print(f"   Size: {info['size_mb']} MB\n")

        print("2. Extracting audio...")
        audio_path = processor.extract_audio(test_video)
        print(f"   Audio saved to: {audio_path}\n")

        print("3. Extracting frames (1 per 2 seconds)...")
        frames = processor.extract_frames(test_video, fps=0.5, max_frames=10)
        print(f"   Extracted {len(frames)} frames")
    else:
        print(f"Test video not found: {test_video}")
