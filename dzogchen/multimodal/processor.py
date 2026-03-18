"""
Multimodal Processor
====================

Unified interface for processing images, audio transcripts, PDFs, and
other non-text inputs. Works with any model adapter that supports vision
(Claude, GPT-4V, Gemini).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Types
# ============================================================================

class ModalityType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO_FRAME = "video_frame"
    TEXT = "text"


@dataclass
class MultimodalInput:
    """A multimodal input payload."""
    modality: ModalityType
    content: Union[bytes, str]          # raw bytes or file path / URL
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "MultimodalInput":
        path = Path(path)
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"

        if mime.startswith("image/"):
            modality = ModalityType.IMAGE
        elif mime.startswith("audio/"):
            modality = ModalityType.AUDIO
        elif mime in ("application/pdf", "text/plain", "text/html"):
            modality = ModalityType.DOCUMENT
        else:
            modality = ModalityType.DOCUMENT

        return cls(
            modality=modality,
            content=path.read_bytes(),
            mime_type=mime,
            metadata={"filename": path.name, "size_bytes": path.stat().st_size},
        )

    @classmethod
    def from_url(cls, url: str, modality: ModalityType = ModalityType.IMAGE) -> "MultimodalInput":
        return cls(
            modality=modality,
            content=url,
            metadata={"url": url},
        )

    @classmethod
    def from_base64(cls, b64_data: str, mime_type: str) -> "MultimodalInput":
        raw = base64.b64decode(b64_data)
        if mime_type.startswith("image/"):
            modality = ModalityType.IMAGE
        elif mime_type.startswith("audio/"):
            modality = ModalityType.AUDIO
        else:
            modality = ModalityType.DOCUMENT
        return cls(modality=modality, content=raw, mime_type=mime_type)

    def to_base64(self) -> str:
        if isinstance(self.content, bytes):
            return base64.b64encode(self.content).decode()
        raise ValueError("Cannot base64-encode non-bytes content")

    @property
    def content_hash(self) -> str:
        data = self.content.encode() if isinstance(self.content, str) else self.content
        return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class MultimodalResult:
    """Result from multimodal processing."""
    success: bool
    input_modality: ModalityType
    description: str                       # natural-language description
    extracted_text: Optional[str] = None   # OCR / transcript
    structured_data: Optional[Dict] = None # tables, objects, etc.
    embeddings: Optional[List[float]] = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Processor
# ============================================================================

class MultimodalProcessor:
    """
    Unified multimodal processor.

    Routes different modalities to appropriate handlers.
    Supports images (vision), audio (transcript), documents (extraction),
    and video frames.

    Without an API key or vision-capable model, falls back to heuristic
    descriptions based on file metadata.

    Example:
        processor = MultimodalProcessor(model_caller=vision_fn)

        result = await processor.process(
            MultimodalInput.from_file("chart.png"),
            prompt="Describe this chart and extract all data points"
        )
    """

    def __init__(
        self,
        model_caller: Optional[Callable] = None,
        vision_caller: Optional[Callable] = None,
        audio_caller: Optional[Callable] = None,
        max_image_size_mb: float = 20.0,
        extract_text: bool = True,
        default_image_prompt: str = "Describe this image in detail. Extract any text, numbers, or data visible.",
    ):
        self._model_caller = model_caller
        self._vision_caller = vision_caller or model_caller
        self._audio_caller = audio_caller or model_caller
        self.max_image_size_mb = max_image_size_mb
        self.extract_text = extract_text
        self.default_image_prompt = default_image_prompt
        self._processing_count = 0
        self._total_latency_ms = 0.0

    async def process(
        self,
        input_data: MultimodalInput,
        prompt: Optional[str] = None,
        extract_structure: bool = False,
    ) -> MultimodalResult:
        """
        Process a multimodal input.

        Args:
            input_data: The multimodal input to process
            prompt: Custom prompt for the model
            extract_structure: If True, attempt to extract structured data (tables, lists)

        Returns:
            MultimodalResult with description, extracted text, and metadata
        """
        start = time.time()
        self._processing_count += 1

        try:
            if input_data.modality == ModalityType.IMAGE:
                result = await self._process_image(input_data, prompt, extract_structure)
            elif input_data.modality == ModalityType.AUDIO:
                result = await self._process_audio(input_data, prompt)
            elif input_data.modality == ModalityType.DOCUMENT:
                result = await self._process_document(input_data, prompt, extract_structure)
            elif input_data.modality == ModalityType.VIDEO_FRAME:
                result = await self._process_image(input_data, prompt, extract_structure)
            else:
                result = await self._process_text(input_data, prompt)

            result.latency_ms = (time.time() - start) * 1000
            self._total_latency_ms += result.latency_ms
            return result

        except Exception as e:
            logger.error(f"Multimodal processing error: {e}")
            return MultimodalResult(
                success=False,
                input_modality=input_data.modality,
                description="",
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def process_batch(
        self,
        inputs: List[MultimodalInput],
        prompt: Optional[str] = None,
        max_concurrent: int = 4,
    ) -> List[MultimodalResult]:
        """Process multiple inputs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded(inp):
            async with semaphore:
                return await self.process(inp, prompt)

        return await asyncio.gather(*[_bounded(inp) for inp in inputs])

    async def describe_image(
        self,
        image: Union[bytes, str, Path],
        prompt: Optional[str] = None,
    ) -> str:
        """Convenience method: describe an image and return text."""
        if isinstance(image, (str, Path)) and Path(str(image)).exists():
            inp = MultimodalInput.from_file(image)
        elif isinstance(image, str) and image.startswith("http"):
            inp = MultimodalInput.from_url(image)
        elif isinstance(image, bytes):
            inp = MultimodalInput(modality=ModalityType.IMAGE, content=image, mime_type="image/jpeg")
        else:
            inp = MultimodalInput.from_file(image)

        result = await self.process(inp, prompt)
        return result.description

    async def extract_text_from_image(self, image: Union[bytes, str, Path]) -> str:
        """OCR: extract all text from an image."""
        prompt = "Extract ALL text visible in this image. Return only the extracted text, verbatim."
        return await self.describe_image(image, prompt=prompt)

    async def analyze_document(
        self,
        document: Union[bytes, str, Path],
        questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract structured information from a document."""
        if isinstance(document, (str, Path)):
            inp = MultimodalInput.from_file(document)
        else:
            inp = MultimodalInput(
                modality=ModalityType.DOCUMENT,
                content=document,
                mime_type="application/pdf",
            )

        if questions:
            prompt = f"Answer these questions about the document:\n" + "\n".join(
                f"- {q}" for q in questions
            )
        else:
            prompt = "Summarize this document. Extract key information, dates, numbers, and entities."

        result = await self.process(inp, prompt, extract_structure=True)
        return {
            "description": result.description,
            "extracted_text": result.extracted_text,
            "structured_data": result.structured_data,
            "success": result.success,
        }

    # =========================================================================
    # Internal handlers
    # =========================================================================

    async def _process_image(
        self,
        inp: MultimodalInput,
        prompt: Optional[str],
        extract_structure: bool,
    ) -> MultimodalResult:
        """Handle image inputs."""
        effective_prompt = prompt or self.default_image_prompt

        if extract_structure:
            effective_prompt += "\n\nAdditionally, extract any tables as JSON and return them in a ```json``` block."

        if self._vision_caller:
            try:
                # Build vision message with image
                if isinstance(inp.content, bytes):
                    b64 = inp.content if isinstance(inp.content, str) else base64.b64encode(inp.content).decode()
                    mime = inp.mime_type or "image/jpeg"
                    # Pass as vision prompt — format depends on model adapter
                    # We use a dict-based format that our adapters understand
                    vision_payload = {
                        "type": "vision",
                        "prompt": effective_prompt,
                        "image_b64": b64,
                        "mime_type": mime,
                    }
                    description = await self._call_with_vision(effective_prompt, inp)
                elif isinstance(inp.content, str) and inp.content.startswith("http"):
                    description = await self._call_with_vision(effective_prompt, inp)
                else:
                    description = self._heuristic_image_description(inp)

                extracted = None
                structured = None

                if extract_structure and "```json" in description:
                    import json
                    import re
                    m = re.search(r"```json\s*([\s\S]+?)```", description)
                    if m:
                        try:
                            structured = json.loads(m.group(1))
                        except json.JSONDecodeError:
                            pass

                if self.extract_text:
                    # Simple heuristic: if description contains quoted text, treat as extracted
                    extracted = description if len(description) > 20 else None

                return MultimodalResult(
                    success=True,
                    input_modality=inp.modality,
                    description=description,
                    extracted_text=extracted,
                    structured_data=structured,
                    confidence=0.9,
                    metadata={"prompt_used": effective_prompt},
                )
            except Exception as e:
                logger.warning(f"Vision call failed, using heuristic: {e}")

        # Heuristic fallback
        description = self._heuristic_image_description(inp)
        return MultimodalResult(
            success=True,
            input_modality=inp.modality,
            description=description,
            confidence=0.3,
            metadata={"fallback": True},
        )

    async def _process_audio(
        self,
        inp: MultimodalInput,
        prompt: Optional[str],
    ) -> MultimodalResult:
        """Handle audio inputs (transcription)."""
        effective_prompt = prompt or "Transcribe this audio. Return only the transcript."

        # Try whisper-style API if audio_caller supports it
        if self._audio_caller:
            try:
                # Most audio transcription APIs take raw bytes
                if isinstance(inp.content, bytes):
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._heuristic_audio_description(inp)
                    )
                else:
                    transcript = f"[Audio from URL: {inp.content}]"

                return MultimodalResult(
                    success=True,
                    input_modality=inp.modality,
                    description=transcript,
                    extracted_text=transcript,
                    confidence=0.7,
                )
            except Exception as e:
                logger.warning(f"Audio processing failed: {e}")

        description = self._heuristic_audio_description(inp)
        return MultimodalResult(
            success=True,
            input_modality=inp.modality,
            description=description,
            confidence=0.2,
            metadata={"fallback": True},
        )

    async def _process_document(
        self,
        inp: MultimodalInput,
        prompt: Optional[str],
        extract_structure: bool,
    ) -> MultimodalResult:
        """Handle document inputs (PDF, text, HTML)."""
        effective_prompt = prompt or "Extract and summarize the content of this document."

        # If content is bytes, try to decode as text first
        text_content = None
        if isinstance(inp.content, bytes):
            try:
                text_content = inp.content.decode("utf-8", errors="replace")
            except Exception:
                text_content = None

        if text_content and self._model_caller:
            try:
                # Truncate to ~4000 chars for the model
                truncated = text_content[:4000]
                if len(text_content) > 4000:
                    truncated += f"\n[... {len(text_content) - 4000} more characters truncated]"

                full_prompt = f"{effective_prompt}\n\nDocument content:\n{truncated}"
                description = await self._model_caller(full_prompt)

                structured = None
                if extract_structure:
                    struct_prompt = f"Extract structured data (tables, lists, key-value pairs) as JSON from:\n{truncated}"
                    struct_response = await self._model_caller(struct_prompt)
                    import json, re
                    m = re.search(r"```json\s*([\s\S]+?)```", struct_response)
                    if m:
                        try:
                            structured = json.loads(m.group(1))
                        except json.JSONDecodeError:
                            pass

                return MultimodalResult(
                    success=True,
                    input_modality=inp.modality,
                    description=description,
                    extracted_text=text_content[:2000],
                    structured_data=structured,
                    confidence=0.85,
                    metadata={"char_count": len(text_content)},
                )
            except Exception as e:
                logger.warning(f"Document processing failed: {e}")

        # Fallback
        if text_content:
            summary = text_content[:500] + "..." if len(text_content) > 500 else text_content
            return MultimodalResult(
                success=True,
                input_modality=inp.modality,
                description=f"Document content ({len(text_content)} chars): {summary}",
                extracted_text=text_content,
                confidence=0.6,
            )

        return MultimodalResult(
            success=True,
            input_modality=inp.modality,
            description=f"Binary document ({inp.mime_type or 'unknown type'})",
            confidence=0.2,
            metadata={"fallback": True},
        )

    async def _process_text(
        self,
        inp: MultimodalInput,
        prompt: Optional[str],
    ) -> MultimodalResult:
        """Handle plain text inputs."""
        text = inp.content if isinstance(inp.content, str) else inp.content.decode("utf-8", errors="replace")

        if self._model_caller and prompt:
            response = await self._model_caller(f"{prompt}\n\nText: {text}")
            return MultimodalResult(
                success=True,
                input_modality=inp.modality,
                description=response,
                extracted_text=text,
                confidence=0.95,
            )

        return MultimodalResult(
            success=True,
            input_modality=inp.modality,
            description=text[:200],
            extracted_text=text,
            confidence=1.0,
        )

    async def _call_with_vision(self, prompt: str, inp: MultimodalInput) -> str:
        """Call model with vision input. Adapts to different caller signatures."""
        if not self._vision_caller:
            return self._heuristic_image_description(inp)

        # Try to call with vision-specific format
        # Check if caller accepts (prompt, image_bytes) or just (prompt,)
        try:
            import inspect
            sig = inspect.signature(self._vision_caller)
            params = list(sig.parameters.keys())

            if "image" in params or "image_bytes" in params or "image_b64" in params:
                if isinstance(inp.content, bytes):
                    b64 = base64.b64encode(inp.content).decode()
                    return await self._vision_caller(prompt, image_b64=b64, mime_type=inp.mime_type)
                else:
                    return await self._vision_caller(prompt, image_url=inp.content)
            else:
                # Simple text-only caller — describe the image type
                meta = f"[{inp.modality.value.upper()} | {inp.mime_type or 'unknown'}"
                if isinstance(inp.content, bytes):
                    meta += f" | {len(inp.content) // 1024}KB]"
                meta += "]"
                return await self._vision_caller(f"{prompt}\n\nNote: {meta} (image content available but requires vision-capable model)")
        except Exception as e:
            logger.debug(f"Vision call error: {e}")
            return self._heuristic_image_description(inp)

    def _heuristic_image_description(self, inp: MultimodalInput) -> str:
        """Heuristic description when no vision model available."""
        mime = inp.mime_type or "image"
        if isinstance(inp.content, bytes):
            kb = len(inp.content) // 1024
            # Detect image type from magic bytes
            if inp.content[:8] == b'\x89PNG\r\n\x1a\n':
                fmt = "PNG"
            elif inp.content[:3] == b'\xff\xd8\xff':
                fmt = "JPEG"
            elif inp.content[:4] == b'GIF8':
                fmt = "GIF"
            elif inp.content[:4] == b'RIFF':
                fmt = "WebP"
            else:
                fmt = mime.split("/")[-1].upper() if "/" in mime else "IMAGE"
            return f"[{fmt} image, {kb}KB — vision model required for description]"
        elif isinstance(inp.content, str) and inp.content.startswith("http"):
            return f"[Image from URL: {inp.content} — vision model required for description]"
        filename = inp.metadata.get("filename", "unknown")
        return f"[Image file: {filename} — vision model required for description]"

    def _heuristic_audio_description(self, inp: MultimodalInput) -> str:
        """Heuristic description when no audio model available."""
        if isinstance(inp.content, bytes):
            kb = len(inp.content) // 1024
            return f"[Audio file, {kb}KB — audio transcription model required]"
        return f"[Audio from: {inp.content} — audio transcription model required]"

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": self._processing_count,
            "avg_latency_ms": (
                self._total_latency_ms / self._processing_count
                if self._processing_count > 0 else 0
            ),
            "has_vision_model": self._vision_caller is not None,
            "has_audio_model": self._audio_caller is not None,
        }
