from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import time
import re

# Import our utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.configs import (
    OUTPUT_DIR, 
    LOGS_DIR,
    TEMP_DIR,
    GROQ_API_KEY_VIDEO,
    SUPPORTED_VIDEO_FORMATS,
    MAX_VIDEO_SIZE,
    FRAME_EXTRACTION_FPS,
    MAX_FRAMES_PER_VIDEO
)
from utils.error_handler import ErrorHandler
from utils.text_cleaner import TextCleaner
from utils.video_processor import VideoProcessor
from utils.ocr_handler import OCRHandler


class VideoExtractor:
    """Extract text from video files using audio transcription and OCR"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.text_cleaner = TextCleaner()
        self.base_output_dir = OUTPUT_DIR
        self.base_logs_dir = LOGS_DIR
        self.temp_dir = TEMP_DIR
        
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_logs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_processor = VideoProcessor()
        self.ocr_handler = OCRHandler()
        
        self.api_key = api_key or GROQ_API_KEY_VIDEO
    
    # ────────────────────────────────────────
    # Existing methods (unchanged except where noted)
    # ────────────────────────────────────────
    
    def _create_resource_name(self, filename: str) -> str:
        name = Path(filename).stem.lower()
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)
        name = name.strip('_')
        return name[:50] or "unnamed_video"
    
    def _setup_resource_directories(self, resource_name: str) -> tuple:
        out_dir = self.base_output_dir / resource_name
        log_dir = self.base_logs_dir / resource_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, log_dir
    
    def _validate_video_file(self, video_path: Path, error_handler: ErrorHandler) -> bool:
        if not video_path.exists():
            error_handler.log_error(FileNotFoundError(f"Video not found: {video_path}"))
            return False
        
        if video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            error_handler.log_error(ValueError(f"Unsupported format: {video_path.suffix}"))
            return False
        
        size_mb = video_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE:
            error_handler.log_error(ValueError(f"File too large: {size_mb:.2f}MB"))
            return False
        
        return True
    
    def _format_timestamp(self, seconds: float) -> str:
        td = timedelta(seconds=int(seconds))
        total = int(td.total_seconds())
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
    
    def _transcribe_with_timestamps(self, audio_path: str, error_handler: ErrorHandler) -> List[Dict]:
        # (unchanged - your original method)
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            with open(audio_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(Path(audio_path).name, f.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            segments = []
            if hasattr(transcription, 'segments') and transcription.segments:
                for seg in transcription.segments:
                    if isinstance(seg, dict):
                        segments.append({
                            'start': seg.get('start', 0),
                            'end': seg.get('end', 0),
                            'text': seg.get('text', '').strip()
                        })
                    else:
                        segments.append({
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text.strip()
                        })
            else:
                text = getattr(transcription, 'text', str(transcription))
                segments.append({'start': 0, 'end': 0, 'text': text})
            
            return segments
            
        except Exception as e:
            error_handler.log_error(e, context="Audio transcription")
            return []
    
    def _extract_frames_with_timestamps(self, video_path: str, resource_name: str,
                                       fps: float, max_frames: int,
                                       error_handler: ErrorHandler) -> List[Dict]:
        # (unchanged)
        try:
            frame_paths = self.video_processor.extract_frames(
                video_path,
                str(self.temp_dir / f"{resource_name}_frames"),
                fps=fps,
                max_frames=max_frames
            )
            if not frame_paths:
                return []
            
            frames_data = []
            interval = 1.0 / fps
            
            for i, path in enumerate(frame_paths):
                ts = i * interval
                ocr = self.ocr_handler.extract_text_from_image(
                    path,
                    detect_equations=True,
                    confidence_threshold=0.5
                )
                frames_data.append({
                    'timestamp': ts,
                    'frame_number': i + 1,
                    'path': path,
                    'text': ocr['text'],
                    'equations': ocr['equations']
                })
            return frames_data
        except Exception as e:
            error_handler.log_error(e, context="Frame extraction/OCR")
            return []
    
    def _organize_by_timeline(self, audio_segments: List[Dict], frames_data: List[Dict],
                             error_handler: ErrorHandler) -> List[Dict]:
        # (unchanged)
        timeline = []
        
        if not audio_segments and not frames_data:
            return timeline
        
        if audio_segments and not frames_data:
            for seg in audio_segments:
                timeline.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'audio': seg['text'],
                    'frames': []
                })
            return timeline
        
        if frames_data and not audio_segments:
            for f in frames_data:
                timeline.append({
                    'start': f['timestamp'],
                    'end': f['timestamp'] + 1,
                    'audio': '',
                    'frames': [f]
                })
            return timeline
        
        for seg in audio_segments:
            s, e = seg['start'], seg['end']
            matching = [f for f in frames_data if s <= f['timestamp'] <= e]
            timeline.append({
                'start': s,
                'end': e,
                'audio': seg['text'],
                'frames': matching
            })
        
        return timeline
    
    def _format_timeline_output(self, timeline: List[Dict]) -> tuple:
        # (unchanged - your original detailed format)
        combined_parts = []
        all_equations = []
        all_audio = []
        all_ocr = []
        
        for i, segment in enumerate(timeline, 1):
            start_ts = self._format_timestamp(segment['start'])
            end_ts = self._format_timestamp(segment['end'])
            
            header = f"\n{'='*70}\nSEGMENT {i} - [{start_ts} → {end_ts}]\n{'='*70}\n\n"
            combined_parts.append(header)
            
            if segment['audio']:
                combined_parts.append(f"🎤 AUDIO TRANSCRIPTION:\n{segment['audio']}\n\n")
                all_audio.append(f"[{start_ts}] {segment['audio']}")
            
            if segment['frames']:
                for frame in segment['frames']:
                    frame_ts = self._format_timestamp(frame['timestamp'])
                    if frame['text']:
                        combined_parts.append(
                            f"📺 SCREEN TEXT (Frame {frame['frame_number']} at {frame_ts}):\n{frame['text']}\n\n"
                        )
                        all_ocr.append(f"[{frame_ts}] {frame['text']}")
                    
                    if frame['equations']:
                        combined_parts.append(
                            f"📐 MATHEMATICAL EQUATIONS (Frame {frame['frame_number']} at {frame_ts}):\n{frame['equations']}\n\n"
                        )
                        all_equations.append(f"[{frame_ts}] {frame['equations']}")
        
        return (
            ''.join(combined_parts),
            '\n\n'.join(all_equations) if all_equations else "",
            '\n\n'.join(all_audio) if all_audio else "",
            '\n\n'.join(all_ocr) if all_ocr else ""
        )
    
    def _format_clean_transcript(self, timeline: List[Dict]) -> str:
        # (unchanged)
        paragraphs = []
        current = []
        last_end = 0.0

        for segment in timeline:
            audio = segment.get('audio', '').strip()
            if not audio or len(audio) < 15:
                continue

            start = segment['start']
            if start - last_end > 5.0 and current:
                paragraphs.append(" ".join(current))
                current = []

            ts_prefix = ""
            if not current or (start - last_end > 5.0):
                ts_prefix = f"[{self._format_timestamp(start)}] "

            cleaned = re.sub(r'\s+', ' ', audio)
            cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned)
            cleaned = cleaned.strip()

            if cleaned:
                current.append(cleaned)

            last_end = segment['end']

        if current:
            paragraphs.append(" ".join(current))

        full = "\n\n".join(paragraphs)
        full = re.sub(r'\s+', ' ', full).strip()
        full = full.replace(' . ', '. ').replace(' , ', ', ')
        return full
    
    # ────────────────────────────────────────
    # NEW: Parsing helpers (from your combine script)
    # ────────────────────────────────────────
    
    def _parse_audio_transcript_file(self, file_path: Path) -> List[Dict]:
        entries = []
        pattern = r'\[(\d{2}:\d{2}(?::\d{2})?)\]\s*(.+)'
        
        if not file_path.is_file():
            return entries
            
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(pattern, line)
                if match:
                    ts_str, text = match.groups()
                    seconds = self._timestamp_to_seconds(ts_str)
                    entries.append({
                        'start_sec': seconds,
                        'end_sec': None,
                        'text': text.strip()
                    })
        return entries
    
    def _parse_screen_text_file(self, file_path: Path) -> List[Dict]:
        entries = []
        pattern = r'\[(\d{2}:\d{2})\]\s*(.+)'
        
        if not file_path.is_file():
            return entries
            
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(pattern, line)
                if match:
                    ts_str, text = match.groups()
                    seconds = self._timestamp_to_seconds(ts_str)
                    entries.append({
                        'timestamp_sec': seconds,
                        'text': text.strip()
                    })
        return entries
    
    def _timestamp_to_seconds(self, ts: str) -> float:
        parts = list(map(int, ts.split(':')))
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return 0.0
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"
    
    def _assign_end_times(self, audio_entries: List[Dict]):
        for i in range(len(audio_entries) - 1):
            audio_entries[i]['end_sec'] = audio_entries[i + 1]['start_sec']
        if audio_entries:
            last = audio_entries[-1]
            last['end_sec'] = last['start_sec'] + 6.0  # average sentence
    
    # ────────────────────────────────────────
    # Updated synchronized view (now supports fallback)
    # ────────────────────────────────────────
    
    def _format_synchronized_view(self, timeline: List[Dict], output_dir: Path, resource_name: str) -> str:
        lines = []
        
        # Try in-memory timeline first (preferred)
        has_audio = any(bool(seg.get('audio', '').strip()) for seg in timeline)
        
        if has_audio:
            # Group by audio segments
            for seg in timeline:
                spoken = seg.get('audio', '').strip()
                if not spoken:
                    continue
                    
                start_str = self._format_timestamp(seg['start'])
                end_str   = self._format_timestamp(seg['end'])
                
                lines.append(f"[{start_str} – {end_str}]")
                lines.append(f"Spoken:  {spoken}")
                lines.append("")
                
                screen_set = set()
                for frame in seg['frames']:
                    txt = frame['text'].strip()
                    if len(txt) >= 10:
                        screen_set.add(txt)
                
                if screen_set:
                    lines.append("On screen:")
                    for txt in sorted(screen_set):
                        lines.append(f"- {txt}")
                else:
                    lines.append("(no significant on-screen text)")
                lines.append("")
        
        else:
            # Fallback: read saved files (your current situation)
            audio_file = output_dir / f"{resource_name}_audio_transcript.txt"
            screen_file = output_dir / f"{resource_name}_screen_text.txt"
            
            if not audio_file.is_file() and not screen_file.is_file():
                return "No audio or screen text files found for synchronization."
            
            audio_data = self._parse_audio_transcript_file(audio_file)
            screen_data = self._parse_screen_text_file(screen_file)
            
            if audio_data:
                self._assign_end_times(audio_data)
                # Use parsed audio + screen
                screen_sorted = sorted(screen_data, key=lambda x: x['timestamp_sec'])
                
                for entry in audio_data:
                    start = entry['start_sec']
                    end = entry.get('end_sec', start + 8.0)
                    
                    lines.append(f"[{self._seconds_to_timestamp(start)} – {self._seconds_to_timestamp(end)}]")
                    lines.append(f"Spoken:  {entry['text']}")
                    lines.append("")
                    
                    visible = set()
                    for s in screen_sorted:
                        if start <= s['timestamp_sec'] < end:
                            txt = s['text'].strip()
                            if len(txt) >= 10:
                                visible.add(txt)
                    
                    if visible:
                        lines.append("On screen:")
                        for txt in sorted(visible):
                            lines.append(f"- {txt}")
                    else:
                        lines.append("(no significant on-screen text)")
                    lines.append("")
            
            elif screen_data:
                # Only screen text → group by change points
                lines.append("No spoken audio detected. Grouping by screen text changes.")
                current_text = None
                block_start = None
                block_texts = []
                
                screen_sorted = sorted(screen_data, key=lambda x: x['timestamp_sec'])
                
                for entry in screen_sorted:
                    txt = entry['text'].strip()
                    if len(txt) < 10:
                        continue
                        
                    t = entry['timestamp_sec']
                    
                    if txt != current_text:
                        if current_text and block_start is not None:
                            start_str = self._seconds_to_timestamp(block_start)
                            end_str   = self._seconds_to_timestamp(t)
                            lines.append(f"[{start_str} – {end_str}]")
                            lines.append("(no spoken audio)")
                            lines.append("")
                            lines.append("On screen:")
                            for bt in sorted(set(block_texts)):
                                lines.append(f"- {bt}")
                            lines.append("")
                        
                        block_start = t
                        block_texts = [txt]
                        current_text = txt
                    else:
                        if txt not in block_texts:
                            block_texts.append(txt)
                
                # Last block
                if current_text and block_start is not None:
                    start_str = self._seconds_to_timestamp(block_start)
                    end_str   = self._seconds_to_timestamp(screen_sorted[-1]['timestamp_sec'] + 5)
                    lines.append(f"[{start_str} – {end_str}]")
                    lines.append("(no spoken audio)")
                    lines.append("")
                    lines.append("On screen:")
                    for bt in sorted(set(block_texts)):
                        lines.append(f"- {bt}")
                    lines.append("")
            
            else:
                lines.append("No usable audio or screen text content found.")
        
        return "\n".join(lines) if lines else "No synchronized content available."
    
    def extract(self, 
                video_path: str,
                resource_id: Optional[str] = None,
                clean_text: bool = True,
                extract_audio: bool = True,
                extract_frames: bool = True,
                fps: Optional[float] = None,
                max_frames: Optional[int] = None) -> Dict[str, Any]:
        start_time = time.time()
        video_path = Path(video_path)
        
        resource_name = self._create_resource_name(video_path.name)
        output_dir, logs_dir = self._setup_resource_directories(resource_name)
        
        error_handler = ErrorHandler(f"video_{resource_name}")
        error_handler.log_file = logs_dir / "extraction.log"
        error_handler.logger = error_handler._setup_logger()
        
        if not self._validate_video_file(video_path, error_handler):
            return self._create_error_result(resource_name, "Validation failed", output_dir, video_path.name)
        
        file_size = video_path.stat().st_size
        
        try:
            video_info = self.video_processor.get_video_info(str(video_path))
            
            audio_segments = []
            frames_data = []
            
            if extract_audio:
                audio_path = self.video_processor.extract_audio(
                    str(video_path),
                    str(self.temp_dir / f"{resource_name}_audio.wav")
                )
                audio_segments = self._transcribe_with_timestamps(audio_path, error_handler)
                Path(audio_path).unlink(missing_ok=True)
            
            if extract_frames:
                fps_val = fps or FRAME_EXTRACTION_FPS
                max_f = max_frames or MAX_FRAMES_PER_VIDEO
                frames_data = self._extract_frames_with_timestamps(
                    str(video_path), resource_name, fps_val, max_f, error_handler
                )
            
            timeline = self._organize_by_timeline(audio_segments, frames_data, error_handler)
            
            if not timeline:
                raise ValueError("No content extracted")
            
            combined_text, equations_text, audio_text, ocr_text = self._format_timeline_output(timeline)
            
            if clean_text:
                combined_text = self.text_cleaner.clean_text(
                    combined_text, remove_urls=False, remove_emails=False, fix_spacing=True
                )
            
            clean_transcript = self._format_clean_transcript(timeline)
            
            # ── Synchronized view (now always attempts fallback) ──
            sync_view = self._format_synchronized_view(timeline, output_dir, resource_name)
            
            # ── Save files ──
            text_file = output_dir / f"{resource_name}_text.txt"
            text_file.write_text(combined_text, encoding='utf-8')
            
            clean_file = output_dir / f"{resource_name}_clean_transcript.txt"
            clean_file.write_text(clean_transcript, encoding='utf-8')
            
            sync_file = output_dir / f"{resource_name}_synchronized.txt"
            sync_file.write_text(sync_view, encoding='utf-8')
            
            equations_file = audio_file = ocr_file = None
            
            if equations_text and equations_text.strip():
                equations_file = output_dir / f"{resource_name}_equations.txt"
                equations_file.write_text(equations_text, encoding='utf-8')
            
            if audio_text and audio_text.strip():
                audio_file = output_dir / f"{resource_name}_audio_transcript.txt"
                audio_file.write_text(audio_text, encoding='utf-8')
            
            if ocr_text and ocr_text.strip():
                ocr_file = output_dir / f"{resource_name}_screen_text.txt"
                ocr_file.write_text(ocr_text, encoding='utf-8')
            
            metadata_file = output_dir / f"{resource_name}_metadata.json"
            
            metadata = {
                "resource_name": resource_name,
                "resource_id": resource_id or resource_name,
                "filename": video_path.name,
                "source_type": "video",
                "video_format": video_path.suffix.lower(),
                "upload_date": datetime.now().isoformat(),
                "extraction_timestamp": datetime.now().isoformat(),
                "file_size_bytes": file_size,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "status": "success",
                "error_message": None,
                "video_info": video_info,
                "timeline_segments": len(timeline),
                "audio_segments": len(audio_segments),
                "frames_processed": len(frames_data),
                "extracted_text_path": str(text_file),
                "clean_transcript_path": str(clean_file),
                "synchronized_view_path": str(sync_file),
                "audio_transcript_path": str(audio_file) if audio_file else None,
                "screen_text_path": str(ocr_file) if ocr_file else None,
                "equations_file_path": str(equations_file) if equations_file else None,
                "metadata_file": str(metadata_file),
            }
            
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            return {
                "success": True,
                "resource_name": resource_name,
                "text_file": str(text_file),
                "clean_transcript_file": str(clean_file),
                "synchronized_file": str(sync_file),
                "equations_file": str(equations_file) if equations_file else None,
                "audio_transcript_file": str(audio_file) if audio_file else None,
                "screen_text_file": str(ocr_file) if ocr_file else None,
                "metadata_file": str(metadata_file),
                "metadata": metadata,
                "extracted_text": combined_text,
                "clean_transcript": clean_transcript,
                "synchronized_view": sync_view,
                "audio_text": audio_text,
                "ocr_text": ocr_text,
                "equations_text": equations_text,
                "timeline": timeline
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_handler.log_error(e, context="Video extraction")
            return self._create_error_result(
                resource_name, str(e), output_dir, video_path.name, file_size, processing_time
            )
    
    def _create_error_result(self, resource_name: str, error_message: str,
                            output_dir: Path, filename: str = "unknown",
                            file_size: int = 0, processing_time: float = 0) -> Dict[str, Any]:
        metadata = {
            "resource_name": resource_name,
            "filename": filename,
            "source_type": "video",
            "upload_date": datetime.now().isoformat(),
            "extraction_timestamp": datetime.now().isoformat(),
            "file_size_bytes": file_size,
            "processing_time_seconds": round(processing_time, 2),
            "status": "failed",
            "error_message": error_message
        }
        metadata_file = output_dir / f"{resource_name}_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        return {
            "success": False,
            "resource_name": resource_name,
            "metadata_file": str(metadata_file),
            "metadata": metadata,
            "error": error_message
        }


# Example usage remains the same...


# ── Example / Testing ──
if __name__ == "__main__":
    from utils.file_picker import FilePicker
    
    print("=== Testing Video Extractor with Clean Transcript ===\n")
    
    if GROQ_API_KEY_VIDEO == "your-groq-api-key-here":
        print("⚠️ Groq API key not set — audio transcription will be skipped.")
    
    try:
        extractor = VideoExtractor()
        picker = FilePicker()
        print("Select a video file...")
        test_video = picker.pick_video()
        picker.close()
        
        if test_video:
            print(f"\nProcessing: {Path(test_video).name}")
            result = extractor.extract(
                video_path=test_video,
                clean_text=True,
                extract_audio=True,
                extract_frames=True
            )
            
            if result['success']:
                print("\nSuccess!")
                print(f"Clean transcript saved to: {result.get('clean_transcript_file')}")
                print(f"Preview (first 400 chars):\n{result['clean_transcript'][:400]}...")
                # ... other prints as before ...
            else:
                print(f"Failed: {result['error']}")
        else:
            print("No file selected.")
            
    except Exception as e:
        print(f"Unexpected error: {e}")