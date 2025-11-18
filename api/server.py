"""
FastAPI server for Svara TTS API.

Provides ElevenLabs-style text-to-speech endpoints with support for
Indian language voices and streaming audio generation.
"""
from __future__ import annotations
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from fastapi.responses import StreamingResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from tts_engine.voice_config import get_all_voices
from tts_engine.orchestrator import SvaraTTSOrchestrator
from tts_engine.timing import get_timing_stats, reset_timing_stats
from tts_engine.utils import load_audio_from_bytes, svara_zero_shot_prompt, svara_prompt
from tts_engine.snac_codec import SNACCodec
from api.models import (
    VoiceResponse,
    VoicesResponse,
    TTSRequest,
    VoiceCloneRequest,
    VoiceCloneResponse,
)


# ============================================================================
# Configuration
# ============================================================================

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")
TTS_DEVICE = os.getenv("TTS_DEVICE", None)  # None = auto-detect (CUDA/MPS/CPU)
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "2048"))
MAX_GENERATION_TOKENS = int(os.getenv("TTS_MAX_TOKENS", str(VLLM_MAX_MODEL_LEN)))

# Global instances (initialized in lifespan)
orchestrator: Optional[SvaraTTSOrchestrator] = None
tokenizer = None  # For zero-shot voice cloning
voice_clone_cache: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global orchestrator, tokenizer
    
    print(f"🚀 Initializing Svara TTS API...")
    print(f"   vLLM URL: {VLLM_BASE_URL}")
    print(f"   Model: {VLLM_MODEL}")
    print(f"   Device: {TTS_DEVICE or 'auto-detect'}")
    
    # Initialize orchestrator with default settings
    # We'll create new instances per request with specific voice settings
    orchestrator = SvaraTTSOrchestrator(
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL,
        speaker_id="English (Male)",  # Default, will be overridden per request
        device=TTS_DEVICE,
        prebuffer_seconds=0.5,
        concurrent_decode=True,
        max_workers=2,
    )
    
    # Load tokenizer for zero-shot voice cloning
    print(f"📦 Loading tokenizer for {VLLM_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL)
    print(f"✓ Tokenizer loaded")
    
    print(f"✓ Orchestrator initialized")
    print(f"✓ Loaded {len(get_all_voices())} voices")
    
    yield
    
    print("🛑 Shutting down Svara TTS API...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Svara TTS API",
    description="Text-to-speech API for Indian languages with streaming support",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "model": VLLM_MODEL,
        "vllm_url": VLLM_BASE_URL,
    }


@app.get("/v1/voices", response_model=VoicesResponse)
async def get_voices(model_id: Optional[str] = None):
    """
    Get list of available voices.
    
    Args:
        model_id: Optional filter by model ID (e.g., "svara-tts-v1")
    
    Returns:
        List of available voices with metadata
    """
    voices = get_all_voices(model_id=model_id)
    return VoicesResponse(
        voices=[VoiceResponse(**voice.to_dict()) for voice in voices]
    )


@app.post("/v1/text-to-speech")
async def text_to_speech(
    # Accept both JSON (via TTSRequest) and multipart/form-data
    text: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
    reference_transcript: Optional[str] = Form(None),
    voice_clone_id: Optional[str] = Form(None),
    voice_clone_tokens: Optional[str] = Form(None),
    model_id: str = Form(default="svara-tts-v1"),
    stream: bool = Form(default=True),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None),
    # JSON body (used when Content-Type is application/json)
    json_body: Optional[TTSRequest] = None,
):
    """
    Convert text to speech with streaming or non-streaming response.
    
    Supports two modes:
    1. Standard TTS: Provide 'voice' parameter
    2. Zero-shot cloning: Provide 'reference_audio' file (and optionally 'reference_transcript')
    
    Accepts both:
    - JSON (Content-Type: application/json) with base64-encoded reference_audio
    - Multipart form data (Content-Type: multipart/form-data) with file upload
    
    Returns:
        Raw PCM16 audio bytes (streaming or complete)
    """
    # Handle both JSON and multipart/form-data
    if json_body is not None:
        # JSON request
        request_text = json_body.text
        request_voice = json_body.voice
        request_reference_audio_bytes = json_body.reference_audio
        request_reference_transcript = json_body.reference_transcript
        request_model_id = json_body.model_id
        request_stream = json_body.stream
        request_temperature = json_body.temperature
        request_top_p = json_body.top_p
        request_top_k = json_body.top_k
        request_repetition_penalty = json_body.repetition_penalty
        request_max_tokens = json_body.max_tokens
        request_voice_clone_id = json_body.voice_clone_id
        request_voice_clone_tokens = json_body.voice_clone_tokens
    else:
        # Multipart form data request
        if text is None:
            raise HTTPException(status_code=400, detail="'text' field is required")
        request_text = text
        request_voice = voice
        request_reference_transcript = reference_transcript
        request_model_id = model_id
        request_stream = stream
        request_temperature = temperature
        request_top_p = top_p
        request_top_k = top_k
        request_repetition_penalty = repetition_penalty
        request_max_tokens = max_tokens
        
        # Handle file upload for reference_audio
        if reference_audio is not None:
            request_reference_audio_bytes = await reference_audio.read()
        else:
            request_reference_audio_bytes = None
        
        if voice_clone_tokens:
            try:
                parsed_clone_tokens = json.loads(voice_clone_tokens)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"voice_clone_tokens must be JSON list of integers: {exc}"
                )
            if not isinstance(parsed_clone_tokens, list) or not all(isinstance(i, int) for i in parsed_clone_tokens):
                raise HTTPException(
                    status_code=400,
                    detail="voice_clone_tokens must be JSON list of integers"
                )
            request_voice_clone_tokens = parsed_clone_tokens
        else:
            request_voice_clone_tokens = None
        request_voice_clone_id = voice_clone_id
    
    # Validate that either preset voice or cloning data is provided
    cloning_inputs = any([
        request_reference_audio_bytes,
        request_voice_clone_tokens,
        request_voice_clone_id,
    ])
    if json_body is not None:
        cloning_inputs = cloning_inputs or bool(json_body.voice_clone_tokens or json_body.voice_clone_id)
    if not request_voice and not cloning_inputs:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'voice' or voice cloning data (reference_audio / voice_clone_id / voice_clone_tokens)"
        )
    
    # Currently only v1 is implemented
    # TODO: Implement other models when svara-tts-v2 is released
    # if request_model_id != "svara-tts-v1":
    #     raise HTTPException(
    #         status_code=501,
    #         detail=f"Model '{request_model_id}' is not yet implemented. Currently only 'svara-tts-v1' is supported."
    #     )
    
    # Resolve voice cloning sources into audio tokens
    audio_tokens: Optional[List[int]] = None
    if request_voice_clone_tokens:
        audio_tokens = request_voice_clone_tokens
    if audio_tokens is None and request_voice_clone_id:
        cache_entry = voice_clone_cache.get(request_voice_clone_id)
        if cache_entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"voice_clone_id '{request_voice_clone_id}' not found or expired"
            )
        audio_tokens = cache_entry["audio_tokens"]
        if not request_reference_transcript and cache_entry.get("reference_transcript"):
            request_reference_transcript = cache_entry["reference_transcript"]
    if audio_tokens is None and request_reference_audio_bytes is not None:
        logger.info(f"Loading reference audio from bytes ({len(request_reference_audio_bytes)} bytes)")
        audio_tensor, sample_rate = load_audio_from_bytes(request_reference_audio_bytes, device=TTS_DEVICE)
        logger.info(f"Audio loaded: shape={audio_tensor.shape}, sr={sample_rate}Hz, min={audio_tensor.min():.3f}, max={audio_tensor.max():.3f}")
        codec = SNACCodec(device=TTS_DEVICE)
        audio_tokens = codec.encode_audio(audio_tensor, input_sample_rate=sample_rate, add_token_offsets=True)
        logger.info(f"Audio tokens encoded to {len(audio_tokens)} tokens")
        logger.info(f"First 10 tokens: {audio_tokens[:10]}")
        logger.info(f"Last 10 tokens: {audio_tokens[-10:]}")

    zero_shot_mode = audio_tokens is not None
    prompt = None
    
    if zero_shot_mode:
        # Build zero-shot prompt (returns token IDs directly)
        prompt = svara_zero_shot_prompt(
            text=request_text,
            audio_tokens=audio_tokens,
            transcript=request_reference_transcript,
            tokenizer=tokenizer
        )
        if isinstance(prompt, list):
            logger.info(f"Prompt built: {len(prompt)} token IDs")
            logger.info(f"Token ID preview (first 50): {prompt[:50]}")
            logger.info(f"Token ID preview (last 50): {prompt[-50:]}")
        else:
            logger.info(f"Prompt built (length: {len(prompt)} chars)")
            logger.info(f"Prompt preview (first 500 chars): {prompt[:500]}")
            logger.info(f"Prompt preview (last 200 chars): {prompt[-200:]}")
    else:
        # Standard TTS mode - build standard prompt
        if not request_voice:
            raise HTTPException(
                status_code=400,
                detail="'voice' parameter is required for standard TTS mode"
            )
        
        prompt = svara_prompt(request_text, request_voice)
        logger.info(f"Standard prompt built (length: {len(prompt)} chars)")
        logger.info(f"Prompt: {prompt}")
    
    # Estimate prompt token count to keep generation within context window
    if isinstance(prompt, list):
        prompt_token_count = len(prompt)
    else:
        if tokenizer is None:
            raise HTTPException(
                status_code=500,
                detail="Tokenizer is not initialized"
            )
        try:
            prompt_token_count = tokenizer(
                prompt,
                add_special_tokens=True,
                return_tensors="pt"
            ).input_ids.shape[-1]
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to tokenize prompt for length estimation: {exc}"
            ) from exc
    
    if prompt_token_count >= VLLM_MAX_MODEL_LEN:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt is too long ({prompt_token_count} tokens) for the "
                f"model context window ({VLLM_MAX_MODEL_LEN}). Please shorten "
                f"the text input."
            )
        )
    
    available_generation_tokens = VLLM_MAX_MODEL_LEN - prompt_token_count
    if available_generation_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="No room left in the context window after prompt tokens."
        )
    
    logger.info(
        "Prompt tokens: %s, available generation tokens: %s (context=%s)",
        prompt_token_count,
        available_generation_tokens,
        VLLM_MAX_MODEL_LEN,
    )
    
    # Use global orchestrator (already initialized, SNAC model cached)
    request_orchestrator = orchestrator
    
    # Build generation kwargs from request parameters
    gen_kwargs = {}
    if request_temperature is not None:
        gen_kwargs["temperature"] = request_temperature
    if request_top_p is not None:
        gen_kwargs["top_p"] = request_top_p
    if request_top_k is not None:
        gen_kwargs["top_k"] = request_top_k
    if request_repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = request_repetition_penalty
    max_tokens_limit = max(1, min(MAX_GENERATION_TOKENS, available_generation_tokens))
    if request_max_tokens is not None:
        effective_max_tokens = min(request_max_tokens, max_tokens_limit)
        if effective_max_tokens < request_max_tokens:
            logger.warning(
                "Requested max_tokens=%s exceeds limit (%s); clamping to %s",
                request_max_tokens,
                max_tokens_limit,
                effective_max_tokens,
            )
        gen_kwargs["max_tokens"] = effective_max_tokens
    else:
        gen_kwargs["max_tokens"] = max_tokens_limit
    
    # Handle streaming vs non-streaming
    if request_stream:
        # Streaming response
        async def audio_stream():
            """Stream audio chunks as they're generated."""
            try:
                async for chunk in request_orchestrator.astream(request_text, prompt=prompt, **gen_kwargs):
                    yield chunk
            except Exception as e:
                print(f"Error during streaming: {e}")
                raise
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "X-Sample-Rate": "24000",
                "X-Bit-Depth": "16",
                "X-Channels": "1",
            }
        )
    else:
        # Non-streaming: collect all audio chunks
        try:
            audio_chunks = []
            async for chunk in request_orchestrator.astream(request_text, prompt=prompt, **gen_kwargs):
                audio_chunks.append(chunk)
            
            complete_audio = b"".join(audio_chunks)
            
            return Response(
                content=complete_audio,
                media_type="audio/pcm",
                headers={
                    "Content-Type": "audio/pcm",
                    "X-Sample-Rate": "24000",
                    "X-Bit-Depth": "16",
                    "X-Channels": "1",
                    "Content-Length": str(len(complete_audio)),
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating audio: {str(e)}"
            )


@app.post("/v1/voice-clone", response_model=VoiceCloneResponse)
async def voice_clone_endpoint(
    reference_audio: Optional[UploadFile] = File(None),
    reference_transcript: Optional[str] = Form(None),
    return_tokens: bool = Form(True),
    model_id: str = Form(default="svara-tts-v1"),
    json_body: Optional[VoiceCloneRequest] = None,
):
    """
    Encode reference audio into reusable SNAC audio tokens for zero-shot cloning.
    """
    if json_body is not None:
        audio_bytes = json_body.reference_audio
        transcript = json_body.reference_transcript
        model_id = json_body.model_id
        include_tokens = json_body.return_tokens
    else:
        if reference_audio is None:
            raise HTTPException(
                status_code=400,
                detail="reference_audio file is required"
            )
        audio_bytes = await reference_audio.read()
        transcript = reference_transcript
        include_tokens = return_tokens

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="reference_audio cannot be empty"
        )

    audio_tensor, sample_rate = load_audio_from_bytes(audio_bytes, device=TTS_DEVICE)
    codec = SNACCodec(device=TTS_DEVICE)
    audio_tokens = codec.encode_audio(
        audio_tensor, input_sample_rate=sample_rate, add_token_offsets=True
    )
    voice_id = uuid4().hex
    voice_clone_cache[voice_id] = {
        "audio_tokens": audio_tokens,
        "reference_transcript": transcript,
        "sample_rate": sample_rate,
        "model_id": model_id,
    }

    token_preview = audio_tokens[: min(16, len(audio_tokens))]
    response_payload = VoiceCloneResponse(
        voice_id=voice_id,
        audio_token_count=len(audio_tokens),
        sample_rate_hz=sample_rate,
        transcript_provided=bool(transcript),
        token_preview=token_preview,
        voice_clone_tokens=audio_tokens if include_tokens else None,
    )
    return response_payload


# ============================================================================
# Debug Endpoints
# ============================================================================

@app.get("/debug/timing")
async def get_timing():
    """
    Get timing statistics for all tracked functions.
    
    Returns detailed performance metrics including call counts, average times,
    min/max times for each tracked function.
    """
    stats = get_timing_stats()
    
    # Convert to more readable format
    formatted_stats = {}
    for func_name, data in stats.items():
        # Skip functions that haven't been called yet
        if data["count"] == 0:
            continue
            
        avg_time = data["total_time"] / data["count"]
        # Handle inf values (when count is 0)
        min_time = data["min_time"] if data["min_time"] != float('inf') else 0
        max_time = data["max_time"] if data["max_time"] != float('-inf') else 0
        
        formatted_stats[func_name] = {
            "calls": data["count"],
            "total_ms": round(data["total_time"] * 1000, 2),
            "avg_ms": round(avg_time * 1000, 2),
            "min_ms": round(min_time * 1000, 2),
            "max_ms": round(max_time * 1000, 2),
        }
    
    return {
        "timing_stats": formatted_stats,
        "note": "All times in milliseconds"
    }


@app.post("/debug/timing/reset")
async def reset_timing():
    """
    Reset all timing statistics.
    
    Clears all accumulated timing data. Useful for starting fresh measurements.
    """
    reset_timing_stats()
    return {"status": "success", "message": "Timing statistics have been reset"}


# ============================================================================
# Main Entry Point (for local development only)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8080"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print(f"Starting Svara TTS API on {host}:{port}")
    print("Note: For production, use supervisord to manage processes")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

