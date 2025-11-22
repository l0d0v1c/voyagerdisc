#!/usr/bin/env python3
"""
Encode deux images dans un format audio simple et direct.
Format simplifié non compatible avec le Voyager original.
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

# Paramètres simplifiés
SAMPLE_RATE = 384000
FRAME_HEIGHT = 364
TRACES_PER_FRAME = 540

def intensity_to_audio(intensity):
    """Convertit directement l'intensité (0-255) en valeur audio (-1 à 1)"""
    return (intensity / 255.0) * 2.0 - 1.0  # Mapping linéaire simple

def resize_image_with_letterbox(img, target_width, target_height):
    """Redimensionne une image en gardant le ratio et ajoute des bandes noires"""
    from PIL import Image

    original_width, original_height = img.size
    ratio_width = target_width / original_width
    ratio_height = target_height / original_height
    ratio = min(ratio_width, ratio_height)

    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    result = Image.new('L', (target_width, target_height), 0)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    result.paste(img_resized, (x_offset, y_offset))

    return result

def load_image(image_path):
    """Charge une image et la convertit au format requis"""
    if image_path.lower().endswith('.pgm'):
        img_array = load_pgm(image_path)
        try:
            from PIL import Image
            img = Image.fromarray(img_array, mode='L')
        except ImportError:
            print("PIL requis pour le redimensionnement automatique")
            # Si pas de PIL, juste redimensionner brutalement
            if img_array.shape != (FRAME_HEIGHT, TRACES_PER_FRAME):
                print(f"Attention: redimensionnement brutal {img_array.shape} -> {FRAME_HEIGHT}x{TRACES_PER_FRAME}")
                from scipy.ndimage import zoom
                zoom_y = FRAME_HEIGHT / img_array.shape[0]
                zoom_x = TRACES_PER_FRAME / img_array.shape[1]
                return zoom(img_array, (zoom_y, zoom_x))
            return img_array
    else:
        try:
            from PIL import Image
            img = Image.open(image_path)
            img = img.convert('L')
        except ImportError:
            print("PIL non disponible. Utilisez un fichier PGM.")
            return None

    # Redimensionner avec letterbox si nécessaire
    if img.size != (TRACES_PER_FRAME, FRAME_HEIGHT):
        print(f"  Image originale: {img.size}")
        img = resize_image_with_letterbox(img, TRACES_PER_FRAME, FRAME_HEIGHT)
        print(f"  Redimensionnée vers: {img.size}")

    return np.array(img)

def load_pgm(pgm_path):
    """Charge un fichier PGM"""
    with open(pgm_path, 'rb') as f:
        first_line = f.readline().decode().strip()

        if first_line.startswith('P5'):
            parts = first_line.split()
            if len(parts) == 4:
                _, width, height, maxval = parts
                width, height, maxval = int(width), int(height), int(maxval)
            else:
                if len(parts) == 1:
                    line = f.readline().decode().strip()
                    while line.startswith('#'):
                        line = f.readline().decode().strip()
                    width, height = map(int, line.split())
                    maxval = int(f.readline().decode().strip())
                else:
                    raise ValueError("Format PGM invalide")
        else:
            raise ValueError("Format PGM P5 requis")

        data = f.read()
        image = np.frombuffer(data, dtype=np.uint8)
        image = image.reshape((height, width))

        return image

def encode_simple_image(image_data):
    """Encode une image directement en audio - format simple"""
    # Calculer la taille: chaque pixel = 1 échantillon
    total_samples = FRAME_HEIGHT * TRACES_PER_FRAME
    audio_buffer = np.zeros(total_samples)

    sample_idx = 0
    # Parcourir l'image ligne par ligne (comme un scan TV)
    for y in range(FRAME_HEIGHT):
        for x in range(TRACES_PER_FRAME):
            if y < image_data.shape[0] and x < image_data.shape[1]:
                pixel_value = image_data[y, x]
                audio_buffer[sample_idx] = intensity_to_audio(pixel_value)
            sample_idx += 1

    return audio_buffer

def main():
    if len(sys.argv) < 3:
        print("Usage: python encode_simple_stereo.py <image_left> <image_right> [output.wav]")
        print("Encode deux images dans un format audio simple (non compatible Voyager)")
        sys.exit(1)

    left_image_path = sys.argv[1]
    right_image_path = sys.argv[2]
    output_wav = sys.argv[3] if len(sys.argv) > 3 else "simple_stereo.wav"

    # Vérifier l'existence des fichiers
    for path in [left_image_path, right_image_path]:
        if not os.path.exists(path):
            print(f"Erreur: fichier {path} introuvable")
            sys.exit(1)

    print(f"Chargement de l'image gauche: {left_image_path}")
    left_image = load_image(left_image_path)
    if left_image is None:
        print("Erreur lors du chargement de l'image gauche")
        sys.exit(1)

    print(f"Chargement de l'image droite: {right_image_path}")
    right_image = load_image(right_image_path)
    if right_image is None:
        print("Erreur lors du chargement de l'image droite")
        sys.exit(1)

    print("Encodage simple des images en audio stéréo...")

    # Encoder chaque image
    left_audio = encode_simple_image(left_image)
    right_audio = encode_simple_image(right_image)

    # S'assurer que les deux canaux ont la même longueur
    max_length = max(len(left_audio), len(right_audio))
    if len(left_audio) < max_length:
        left_audio = np.pad(left_audio, (0, max_length - len(left_audio)))
    if len(right_audio) < max_length:
        right_audio = np.pad(right_audio, (0, max_length - len(right_audio)))

    # Créer audio stéréo
    stereo_audio = np.column_stack((left_audio, right_audio))

    print(f"Sauvegarde dans {output_wav}...")
    # Normaliser et convertir en int16
    stereo_audio = np.clip(stereo_audio, -1.0, 1.0)
    audio_int16 = (stereo_audio * 32767).astype(np.int16)

    wavfile.write(output_wav, SAMPLE_RATE, audio_int16)

    print(f"Fichier WAV simple créé: {output_wav}")
    print(f"Durée: {len(stereo_audio)/SAMPLE_RATE:.2f}s")
    print(f"Format: {SAMPLE_RATE}Hz, stéréo, 2 images (left/right)")
    print("Note: Format simple, non compatible avec l'extracteur Voyager original")

if __name__ == "__main__":
    main()