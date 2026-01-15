#!/usr/bin/env python3
"""
Ghost Layers - AI-resistant audio processing for vocal obfuscation.
Evade lyrics detection while preserving melody and singing style.

Usage:
    python ghost_layers.py input.wav --output output.wav --cutoff 1500
"""

import os
import argparse
import subprocess
import logging
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def separate_audio(input_file, output_dir):
    """Separates audio using Demucs."""
    logging.info(f"Separating sources for {input_file}...")
    try:
        cmd = ["demucs", "-n", "htdemucs", "--out", output_dir, input_file]
        subprocess.run(cmd, check=True)
        
        track_name = os.path.splitext(os.path.basename(input_file))[0]
        stem_dir = os.path.join(output_dir, "htdemucs", track_name)
        
        return {
            "vocals": os.path.join(stem_dir, "vocals.wav"),
            "drums": os.path.join(stem_dir, "drums.wav"),
            "bass": os.path.join(stem_dir, "bass.wav"),
            "other": os.path.join(stem_dir, "other.wav"),
        }
    except subprocess.CalledProcessError as e:
        logging.error(f"Demucs failed: {e}")
        raise


def apply_lowpass(audio, cutoff, sr):
    """Applies a low-pass filter."""
    if cutoff >= sr / 2:
        return audio
        
    sos = signal.butter(4, cutoff, 'lp', fs=sr, output='sos')
    
    if audio.ndim == 1:
        return signal.sosfiltfilt(sos, audio)
    
    channels = []
    for ch in audio:
        channels.append(signal.sosfiltfilt(sos, ch))
    return np.array(channels)


def phase_randomize(audio, sr, strength=0.15):
    """Randomizes phase - Light version for instrumentals."""
    if audio.ndim == 1:
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        original_phase = np.angle(stft)
        random_phase = np.random.uniform(0, 2*np.pi, stft.shape)
        blended_phase = original_phase * (1 - strength) + random_phase * strength
        new_stft = magnitude * np.exp(1j * blended_phase)
        return librosa.istft(new_stft, length=len(audio))
    
    channels = []
    for ch in audio:
        stft = librosa.stft(ch)
        magnitude = np.abs(stft)
        original_phase = np.angle(stft)
        random_phase = np.random.uniform(0, 2*np.pi, stft.shape)
        blended_phase = original_phase * (1 - strength) + random_phase * strength
        new_stft = magnitude * np.exp(1j * blended_phase)
        channels.append(librosa.istft(new_stft, length=len(ch)))
    return np.array(channels)


def pitch_shift(audio, sr, n_steps=1.5):
    """Pitch shift."""
    logging.info(f"Pitch shifting by {n_steps} semitones...")
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 1:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    channels = []
    for ch in audio:
        channels.append(librosa.effects.pitch_shift(ch, sr=sr, n_steps=n_steps))
    return np.array(channels)


def stereo_swap(audio):
    """Swaps stereo channels."""
    if audio.ndim == 1 or audio.shape[0] != 2:
        return audio
    return np.array([audio[1], audio[0]])


def time_stretch(audio, rate=1.05):
    """Time stretch."""
    logging.info(f"Time stretching by {rate}x...")
    if audio.ndim == 1:
        return librosa.effects.time_stretch(audio, rate=rate)
    channels = []
    for ch in audio:
        channels.append(librosa.effects.time_stretch(ch, rate=rate))
    return np.array(channels)


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val * 0.95
    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Layers v15 - AI-resistant audio processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ghost_layers.py song.wav --output processed.wav --cutoff 1500
  python ghost_layers.py song.wav -o processed.wav --cutoff 1000  # Safer, more muffled
  python ghost_layers.py song.wav -o processed.wav --cutoff 2000  # Riskier, clearer
        """
    )
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("--output", "-o", default="ghost_output.wav", help="Output file path")
    parser.add_argument("--pitch", type=float, default=1.5, help="Global pitch shift (semitones)")
    parser.add_argument("--stretch", type=float, default=1.05, help="Time stretch factor")
    parser.add_argument("--cutoff", type=int, default=1000, 
                        help="LPF Cutoff Hz. Higher=Clearer, Lower=Safer. Sweet spot: 1500")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found: {args.input_file}")
        return

    sr = 44100
    output_dir = "temp_separation"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Separate vocals from instrumentals
        stems = separate_audio(args.input_file, output_dir)
        
        # 2. Load stems
        logging.info("Loading stems...")
        vocals, _ = librosa.load(stems["vocals"], sr=sr, mono=False)
        drums, _ = librosa.load(stems["drums"], sr=sr, mono=False)
        bass, _ = librosa.load(stems["bass"], sr=sr, mono=False)
        other, _ = librosa.load(stems["other"], sr=sr, mono=False)
        
        # Ensure stereo
        if vocals.ndim == 1: vocals = np.array([vocals, vocals])
        if drums.ndim == 1: drums = np.array([drums, drums])
        if bass.ndim == 1: bass = np.array([bass, bass])
        if other.ndim == 1: other = np.array([other, other])
        
        # Combine instrumentals
        min_len = min(drums.shape[1], bass.shape[1], other.shape[1], vocals.shape[1])
        vocals = vocals[:, :min_len]
        instrumental = drums[:, :min_len] + bass[:, :min_len] + other[:, :min_len]
        
        # ============== GHOST LAYERS PROCESSING ==============
        
        # 1. Instrumentals: Light Phase Randomization
        logging.info("Processing Instrumentals...")
        instrumental = phase_randomize(instrumental, sr, strength=0.15)
        
        # 2. Vocals: The Ghost Layers
        logging.info(f"Processing Vocals (LPF Cutoff: {args.cutoff}Hz)...")
        
        # Layer 1: The Muffled Lead (60%)
        # Low Pass removes consonants, keeps vowels/melody
        logging.info(f"  - Layer 1: Muffled Lead (LPF {args.cutoff}Hz)...")
        layer1 = apply_lowpass(vocals, args.cutoff, sr)
        
        # Layer 2: The Shadow (40%)
        # Octave down + darker filter = "ghost" safety net
        logging.info("  - Layer 2: Shadow (Octave Down + LPF 600Hz)...")
        layer2 = pitch_shift(vocals, sr, n_steps=-12.0)
        layer2 = apply_lowpass(layer2, 600, sr)
        
        # Mix Layers (60/40 balance)
        vocal_mix = (layer1 * 0.6) + (layer2 * 0.4)
        vocal_mix = vocal_mix * 1.5  # Moderate boost
        
        # 3. Global Pitch Shift
        logging.info(f"Global Pitch Shift: +{args.pitch} semitones...")
        instrumental = pitch_shift(instrumental, sr, n_steps=args.pitch)
        vocal_mix = pitch_shift(vocal_mix, sr, n_steps=args.pitch)
        
        # Mix
        mixed = instrumental + vocal_mix
        
        # Global Time Stretch
        if args.stretch != 1.0:
            mixed = time_stretch(mixed, rate=args.stretch)
            
        mixed = stereo_swap(mixed)
        final_audio = normalize_audio(mixed)
        
        sf.write(args.output, final_audio.T, sr, subtype='PCM_16')
        
        logging.info(f"✓ Done! Saved to {args.output}")
        logging.info(f"  Config: Ghost Layers (LPF {args.cutoff}Hz / -12st LPF 600Hz)")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
