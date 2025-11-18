#!/bin/bash
# Test script for voice cloning with Svara TTS

API_URL="http://localhost:8080"
REFERENCE_AUDIO="vanshika.wav"

echo "🎤 Step 1: Cloning voice from $REFERENCE_AUDIO..."
VOICE_ID=$(curl -s -X POST "$API_URL/v1/voice-clone" \
  -F "reference_audio=@$REFERENCE_AUDIO" \
  -F "return_tokens=false" | python3 -c "import sys, json; print(json.load(sys.stdin)['voice_id'])")

if [ -z "$VOICE_ID" ]; then
  echo "❌ Failed to clone voice"
  exit 1
fi

echo "✓ Voice cloned! Voice ID: $VOICE_ID"
echo ""

echo "🗣️  Step 2: Generating speech with cloned voice..."
curl -s -X POST "$API_URL/v1/text-to-speech" \
  -F "text=Hello, this is a test of the cloned voice. The voice cloning feature is working perfectly! <fear>" \
  -F "voice_clone_id=$VOICE_ID" \
  -F "stream=false" \
  --output cloned_output.pcm

if [ ! -f "cloned_output.pcm" ] || [ ! -s "cloned_output.pcm" ]; then
  echo "❌ Failed to generate audio"
  exit 1
fi

# Check if it's an error JSON
if file cloned_output.pcm | grep -q "JSON"; then
  echo "❌ Error response:"
  cat cloned_output.pcm
  exit 1
fi

echo "✓ Audio generated! File: cloned_output.pcm"
echo ""

echo "🎵 Step 3: Converting PCM to WAV..."
ffmpeg -f s16le -ar 24000 -ac 1 -i cloned_output.pcm cloned_output.wav -y 2>&1 | grep -E "time=|error|Error" || true

if [ -f "cloned_output.wav" ]; then
  echo "✓ WAV file created: cloned_output.wav"
  echo ""
  echo "🎉 Success! You can now play cloned_output.wav"
else
  echo "❌ Failed to convert to WAV"
  exit 1
fi

