# Quick Start - Unicorn Orator with NPU (Consolidated)

**Date**: October 23, 2025
**Version**: NPU Consolidated v1.0

---

## 🆕 What's New: Consolidated NPU Support

Orator now uses [unicorn-npu-core](https://github.com/Unicorn-Commander/unicorn-npu-core) - a shared library that provides NPU support across all Unicorn projects.

**Benefits**:
- ✅ Single command installation
- ✅ Automatic host system setup
- ✅ Consistent NPU support across projects
- ✅ Easy updates and bug fixes

---

## 🚀 Installation (One Command!)

```bash
bash scripts/install-orator.sh
```

That's it! This script automatically:
1. Installs `unicorn-npu-core` library
2. Sets up NPU on host system (XRT, drivers, permissions)
3. Installs Orator dependencies
4. Offers to download Kokoro TTS models

**Time**: ~5-10 minutes (first time), ~1-2 minutes (if NPU already set up)

---

## 📋 Manual Installation (Step-by-Step)

If you prefer manual control:

### Step 1: Install unicorn-npu-core

```bash
# From GitHub (once published)
pip install git+https://github.com/Unicorn-Commander/unicorn-npu-core.git

# OR from local development
cd /home/ucadmin/UC-1/unicorn-npu-core
pip install -e . --break-system-packages
```

### Step 2: Setup NPU on Host

```bash
# Using Python
python3 -m unicorn_npu.scripts.install_host

# OR directly
cd /home/ucadmin/UC-1/unicorn-npu-core
bash scripts/install-npu-host.sh
```

**Important**: Log out and log back in after this step!

### Step 3: Install Orator

```bash
cd /home/ucadmin/UC-1/Unicorn-Orator/kokoro-tts
pip install -r requirements.npu.txt --break-system-packages
```

### Step 4: Download Kokoro Models (Optional)

```bash
# Models download automatically on first run
# Or force download:
python3 -c "from kokoro_onnx import Kokoro; k = Kokoro('kokoro-v0_19', lang='a')"
```

---

## ⚡ Set NPU to Performance Mode

For best performance, set NPU to performance mode:

```bash
python3 -c "
from unicorn_npu import NPUDevice

npu = NPUDevice()
if npu.is_available():
    npu.set_power_mode('performance')
    print('✅ NPU set to performance mode')
    print(f'Power state: {npu.get_power_state()}')
else:
    print('❌ NPU not available')
"
```

---

## 🎯 Start the Service

```bash
cd /home/ucadmin/UC-1/Unicorn-Orator/kokoro-tts
python3 server.py
```

**Default Ports**:
- Main TTS: **8880** (or 8885)
- Web Interface: http://localhost:8880/web
- Admin Panel: http://localhost:8880/admin

**API Endpoints**:
- `POST /v1/audio/speech` - OpenAI-compatible TTS
- `GET /health` - Health check
- `GET /voices` - List available voices

---

## 🧪 Test the Service

```bash
# Check health
curl http://localhost:8880/health

# Generate speech
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Unicorn Orator!", "voice": "af_bella"}' \
  --output test.wav

# Play the audio
aplay test.wav  # Linux
# or
afplay test.wav  # macOS
```

---

## 🔍 Verify NPU is Working

```python
from unicorn_npu import NPUDevice

npu = NPUDevice()

if npu.is_available():
    info = npu.get_device_info()
    print(f"✅ NPU Available")
    print(f"   Device: {info['device']}")
    print(f"   Type: {info['type']}")
    print(f"   Power: {npu.get_power_state()}")
else:
    print("❌ NPU not available")
```

---

## 📊 Expected Performance

With NPU acceleration:

| Configuration | RTF | Notes |
|--------------|-----|-------|
| **NPU (AMD)** | **0.213** | 13x faster than CPU |
| **NPU + Turbo** | **0.150** | 30% improvement |
| Intel iGPU | 0.280 | Good alternative |
| CPU | 2.500 | Baseline |

**RTF = Real-Time Factor** (lower is better)

**Note**: RTF 0.213 means 1 second of audio is generated in 0.213 seconds!

---

## 🎭 Available Voices

Orator includes 50+ professional voices:

**American**:
- `af_bella`, `af_sarah`, `am_michael`, `am_adam`

**British**:
- `bf_emma`, `bm_george`

**Special**:
- `af_sky`, `am_echo`, `af_heart`

And many more! Check `/voices` endpoint for full list.

---

## 🛠️ Troubleshooting

### NPU Not Detected

```bash
# Check device exists
ls -la /dev/accel/accel0

# Check permissions
groups | grep -E "(render|video)"

# Re-run host setup
python3 -m unicorn_npu.scripts.install_host
```

### Import Errors

```bash
# Reinstall unicorn-npu-core
pip install --force-reinstall unicorn-npu-core
```

### Model Download Issues

```bash
# Manually download Kokoro model
cd kokoro-tts
python3 -c "
from kokoro_onnx import Kokoro
k = Kokoro('kokoro-v0_19', lang='a')
print('✅ Model downloaded')
"
```

### XRT Issues

```bash
# Check XRT installation
/opt/xilinx/xrt/bin/xrt-smi examine

# Verify XRT version
/opt/xilinx/xrt/bin/xrt-smi version
```

---

## 🎬 Voice Acting Tools

Orator includes specialized tool servers:

| Tool | Port | Purpose |
|------|------|---------|
| Dialogue Generator | 13060 | Multi-character conversations |
| Story Narration | 13061 | Kids stories & audiobooks |
| Podcast Creator | 13062 | Professional podcasts |
| Commercial Voiceover | 13063 | Ads & commercials |

**Start tool servers**:
```bash
cd tool-servers
docker-compose up -d
```

---

## 📚 Additional Documentation

- **Core Library**: [unicorn-npu-core README](https://github.com/Unicorn-Commander/unicorn-npu-core)
- **NPU Setup**: `INSTALL-AMD-NPU.md`
- **Voice Acting**: `README.md` (Voice Acting Tools section)
- **Hardware Integration**: `HARDWARE_INTEGRATION_PLAN.md`

---

## 🔄 Updating

### Update unicorn-npu-core

```bash
pip install --upgrade git+https://github.com/Unicorn-Commander/unicorn-npu-core.git
```

### Update Orator

```bash
cd /home/ucadmin/UC-1/Unicorn-Orator
git pull
cd kokoro-tts
pip install -r requirements.npu.txt --break-system-packages
```

---

## 💡 Tips

1. **Always use performance mode** for production:
   ```bash
   python3 -c "from unicorn_npu import NPUDevice; NPUDevice().set_power_mode('performance')"
   ```

2. **Check NPU status before starting**:
   ```bash
   python3 -c "from unicorn_npu import NPUDevice; print(f'NPU: {NPUDevice().is_available()}')"
   ```

3. **Use Docker for production** (see main README.md)

4. **Monitor NPU power state**:
   ```bash
   /opt/xilinx/xrt/bin/xrt-smi examine | grep -i power
   ```

5. **Experiment with voices** - each has unique characteristics!

---

## 🦄 Related Projects

- **unicorn-npu-core**: Core NPU library (shared)
- **Unicorn-Amanuensis**: STT service with NPU
- **whisper_npu_project**: Unicorn Commander (51x realtime!)
- **amd-npu-utils**: NPU development toolkit

---

## 🌐 Web Interface

Access the web interface at **http://localhost:8880/web**

Features:
- 50+ voice selection
- Speed control (0.5x to 2.0x)
- Advanced settings (sample rate, emotion, pitch)
- Real-time hardware status
- Voice acting tool links

---

**Questions?** Check the [main README](README.md) or [INSTALL-AMD-NPU.md](INSTALL-AMD-NPU.md)

**Ready to speak!** 🎙️
