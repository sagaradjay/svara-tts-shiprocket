# Voice Cloning Test Commands

## Step 1: Clone a voice
```bash
curl -X POST http://localhost:8080/v1/voice-clone \
  -F "reference_audio=@nitish.wav" \
  -F "return_tokens=false"
```

This returns a JSON with `voice_id`. Save it for the next step.

## Step 2: Generate speech with cloned voice
```bash
# Replace VOICE_ID with the ID from step 1
curl -X POST http://localhost:8080/v1/text-to-speech \
  -F "text=Hello, this is a test of the cloned voice." \
  -F "voice_clone_id=VOICE_ID" \
  -F "stream=false" \
  --output output.pcm
```

## Step 3: Convert PCM to WAV
```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

## One-liner (all steps combined)
```bash
# Clone voice and extract voice_id
VOICE_ID=$(curl -s -X POST http://localhost:8080/v1/voice-clone \
  -F "reference_audio=@nitish.wav" \
  -F "return_tokens=false" | python3 -c "import sys, json; print(json.load(sys.stdin)['voice_id'])")

# Generate speech
curl -X POST http://localhost:8080/v1/text-to-speech \
  -F "text=Hello, this is a test of the cloned voice." \
  -F "voice_clone_id=$VOICE_ID" \
  -F "stream=false" \
  --output output.pcm

# Convert to WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav -y
```

## Alternative: Use voice_clone_tokens directly
If you want to use the tokens directly instead of caching:

```bash
# Get tokens from clone endpoint
CLONE_RESPONSE=$(curl -s -X POST http://localhost:8080/v1/voice-clone \
  -F "reference_audio=@nitish.wav" \
  -F "return_tokens=true")

# Extract tokens (requires jq or python)
TOKENS=$(echo $CLONE_RESPONSE | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin)['voice_clone_tokens']))")

# Use tokens directly
curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hello test\", \"voice_clone_tokens\": $TOKENS, \"stream\": false}" \
  --output output.pcm
```
