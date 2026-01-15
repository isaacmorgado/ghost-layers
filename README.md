# 👻 Ghost Layers

**AI-resistant audio processing for vocal obfuscation.**

Evade lyrics detection while preserving melody and singing style.

## 🎯 What it Does

Ghost Layers processes audio to make lyrics undetectable by AI systems (like Suno's copyright detection) while keeping the song's vibe, melody, and singing style intact.

### The "Ghost Layers" Technique

1. **Layer 1 (60%)**: Low-pass filtered vocals at tunable cutoff (removes consonants, keeps vowels/melody)
2. **Layer 2 (40%)**: Octave-down "shadow" vocals with darker filtering (adds weight, obscures identity)
3. **Phase randomization** on instrumentals breaks fingerprinting
4. **Global pitch shift** + **time stretch** for additional evasion

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a song (Sweet Spot: 1500Hz cutoff)
python ghost_layers.py song.wav --output processed.wav --cutoff 1500
```

## 📊 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cutoff` | 1000 | LPF frequency (Hz). **Higher = clearer lyrics, lower = safer evasion**. Sweet spot: 1500 |
| `--pitch` | 1.5 | Global pitch shift (semitones) |
| `--stretch` | 1.05 | Time stretch factor |
| `--output` | ghost_output.wav | Output filename |

## 🎚️ Finding Your Sweet Spot

- **Safe Mode** (`--cutoff 1000`): Maximum evasion, very muffled vocals
- **Sweet Spot** (`--cutoff 1500`): Good balance of clarity and evasion
- **Risky Mode** (`--cutoff 2000+`): Clear vocals, may trigger detection

## 📋 Requirements

- Python 3.10+
- FFmpeg (for Demucs audio processing)

## ⚠️ Disclaimer

This tool is for educational and research purposes. Use responsibly and respect copyright laws.
