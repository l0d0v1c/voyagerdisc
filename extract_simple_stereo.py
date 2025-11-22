#!/usr/bin/env python3
"""
Extrait deux images d'un fichier audio WAV stéréo au format simple.
Compatible avec encode_simple_stereo.py
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

# Paramètres
SAMPLE_RATE = 384000
FRAME_HEIGHT = 364
TRACES_PER_FRAME = 540

def audio_to_intensity(audio_value):
    """Convertit une valeur audio (-1 à 1) en intensité (0-255)"""
    # Mapping linéaire inverse
    normalized = (audio_value + 1.0) / 2.0  # -1,1 -> 0,1
    return int(np.clip(normalized * 255, 0, 255))

def decode_simple_audio(audio_buffer):
    """Décode un canal audio en image"""
    expected_samples = FRAME_HEIGHT * TRACES_PER_FRAME

    # S'assurer qu'on a assez de données
    if len(audio_buffer) < expected_samples:
        print(f"  Attention: pas assez de données ({len(audio_buffer)} < {expected_samples})")
        # Compléter avec des zéros
        audio_buffer = np.pad(audio_buffer, (0, expected_samples - len(audio_buffer)))

    # Prendre seulement les échantillons nécessaires
    audio_buffer = audio_buffer[:expected_samples]

    # Créer l'image
    image = np.zeros((FRAME_HEIGHT, TRACES_PER_FRAME), dtype=np.uint8)

    sample_idx = 0
    # Reconstruire l'image ligne par ligne
    for y in range(FRAME_HEIGHT):
        for x in range(TRACES_PER_FRAME):
            if sample_idx < len(audio_buffer):
                audio_value = audio_buffer[sample_idx]
                image[y, x] = audio_to_intensity(audio_value)
                sample_idx += 1

    return image

def save_image(image_data, output_path):
    """Sauvegarde l'image en PGM et PNG"""
    # Sauvegarder en PGM
    with open(output_path.replace('.png', '.pgm'), 'wb') as f:
        f.write(f"P5 {TRACES_PER_FRAME} {FRAME_HEIGHT} 255\n".encode())
        for y in range(FRAME_HEIGHT):
            for x in range(TRACES_PER_FRAME):
                f.write(bytes([image_data[y, x]]))

    # Sauvegarder en PNG si PIL disponible
    try:
        from PIL import Image
        img = Image.fromarray(image_data, mode='L')
        img.save(output_path)
        return True
    except ImportError:
        print("  PIL non disponible - seul PGM sauvé")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_simple_stereo.py <stereo_audio.wav> [output_dir]")
        print("Extrait deux images d'un fichier WAV stéréo au format simple")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_simple"

    if not os.path.exists(input_file):
        print(f"Erreur: fichier {input_file} introuvable")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Chargement de {input_file}...")
    try:
        sample_rate, audio = wavfile.read(input_file)
        print(f"Sample rate: {sample_rate} Hz, durée: {len(audio)/sample_rate:.1f}s")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        sys.exit(1)

    # Normalisation en float [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Vérifier le format stéréo
    if len(audio.shape) == 1:
        print("Erreur: le fichier n'est pas en stéréo")
        sys.exit(1)

    if audio.shape[1] != 2:
        print(f"Erreur: format audio non supporté ({audio.shape[1]} canaux)")
        sys.exit(1)

    # Extraction des canaux
    left = audio[:, 0]
    right = audio[:, 1]

    print(f"Format audio détecté: stéréo ({len(left)} échantillons par canal)")

    # Décodage des images
    channels = [left, right]
    channel_names = ["left", "right"]

    for ch in range(2):
        print(f"\nDécodage canal {channel_names[ch]}...")
        image_data = decode_simple_audio(channels[ch])

        output_path = os.path.join(output_dir, f"{channel_names[ch]}_simple.png")

        if save_image(image_data, output_path):
            print(f"  Image extraite: {output_path}")
        else:
            print(f"  Image extraite: {output_path.replace('.png', '.pgm')}")

    print(f"\nImages extraites dans le dossier: {output_dir}/")
    print("Format simple utilisé - correspondance directe pixel <-> échantillon")

if __name__ == "__main__":
    main()