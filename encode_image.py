#!/usr/bin/env python3
"""
Encode une image au format Voyager Golden Record dans un fichier audio WAV.
L'image doit avoir le format 540x364 pixels (traces x pixels par trace).
"""

import numpy as np
from scipy.io import wavfile
import os
import math
import sys

# Paramètres du format Voyager
SAMPLE_RATE = 384000
FRAME_HEIGHT = 364
TRACES_PER_FRAME = 540

def intensity_to_float(intensity):
    """Convertit une intensité de pixel (0-255) en valeur audio (-0.26 à 0.18)"""
    lower_bound = -0.26
    upper_bound = 0.18

    # Cas spéciaux
    if intensity >= 255:
        return lower_bound
    if intensity <= 0:
        return upper_bound

    # Inverse de la transformation cosinus
    # intensity = 255 - (cos_term + 1.0) * 255.0 / 2.0
    # cos_term = (255 - intensity) * 2.0 / 255.0 - 1.0
    cos_term = (255 - intensity) * 2.0 / 255.0 - 1.0

    # Assurer que cos_term est dans [-1, 1]
    cos_term = max(-1.0, min(1.0, cos_term))

    # t = arccos(cos_term) / π
    t = math.acos(cos_term) / math.pi

    # f = upper_bound - t * (upper_bound - lower_bound)
    f = upper_bound - t * (upper_bound - lower_bound)

    return f

def generate_sync_signal(length):
    """Génère un signal de synchronisation (pic positif suivi d'un creux négatif)"""
    signal = np.zeros(length)

    # Pic positif au début
    peak_length = 90
    for i in range(peak_length):
        signal[i] = 0.8 * math.sin(math.pi * i / peak_length)

    # Creux négatif
    valley_start = peak_length + 10
    valley_length = 90
    for i in range(valley_length):
        if valley_start + i < length:
            signal[valley_start + i] = -0.6 * math.sin(math.pi * i / valley_length)

    return signal

def encode_image_to_audio(image_data):
    """Encode une image (540x364) en signal audio Voyager"""
    # Calculer la taille totale du buffer
    buffer_size = TRACES_PER_FRAME * 3400 * 3 // 2
    audio_buffer = np.zeros(buffer_size)

    current_pos = 0

    for trace_id in range(TRACES_PER_FRAME):
        # Générer le signal de synchronisation
        sync_signal = generate_sync_signal(220)

        # Ajouter le sync signal
        end_pos = min(current_pos + len(sync_signal), len(audio_buffer))
        audio_buffer[current_pos:end_pos] = sync_signal[:end_pos - current_pos]
        current_pos = end_pos

        # Déterminer les paramètres pour cette trace
        if trace_id < 164:
            image_start = current_pos
            image_length = 2680  # 2900 - 220
        else:
            image_start = current_pos
            image_length = 2677  # 2897 - 220

        # Ajustement pour les traces paires
        if trace_id % 2 == 0:
            image_start -= 12
            image_length += 12

        # Encoder les pixels de cette trace
        pixel_width = image_length / FRAME_HEIGHT

        for pixel_id in range(FRAME_HEIGHT):
            if pixel_id < image_data.shape[0] and trace_id < image_data.shape[1]:
                pixel_intensity = image_data[pixel_id, trace_id]
                audio_value = intensity_to_float(pixel_intensity)
            else:
                audio_value = 0.0

            # Remplir plusieurs échantillons avec cette valeur
            pixel_start = int(image_start + pixel_id * pixel_width)
            pixel_end = int(image_start + (pixel_id + 1) * pixel_width)

            if pixel_end > len(audio_buffer):
                break

            audio_buffer[pixel_start:pixel_end] = audio_value

        # Passer à la trace suivante
        current_pos += 3400

    return audio_buffer

def resize_image_with_letterbox(img, target_width, target_height):
    """Redimensionne une image en gardant le ratio et ajoute des bandes noires si nécessaire"""
    from PIL import Image

    original_width, original_height = img.size

    # Calculer le ratio pour utiliser la pleine largeur
    ratio_width = target_width / original_width
    ratio_height = target_height / original_height

    # Utiliser le ratio qui permet de garder l'image dans les limites
    ratio = min(ratio_width, ratio_height)

    # Calculer les nouvelles dimensions
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Redimensionner l'image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Créer une nouvelle image avec la taille cible (noire)
    result = Image.new('L', (target_width, target_height), 0)  # 0 = noir

    # Calculer la position pour centrer l'image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Coller l'image redimensionnée sur l'image noire
    result.paste(img_resized, (x_offset, y_offset))

    return result

def load_image(image_path):
    """Charge une image PGM ou PNG et la convertit au format Voyager"""
    if image_path.lower().endswith('.pgm'):
        img_array = load_pgm(image_path)
        # Convertir en PIL Image pour traitement uniforme
        try:
            from PIL import Image
            img = Image.fromarray(img_array, mode='L')
        except ImportError:
            print("PIL requis pour le redimensionnement automatique")
            return img_array
    else:
        # Utiliser PIL pour les autres formats
        try:
            from PIL import Image
            img = Image.open(image_path)
            img = img.convert('L')  # Convertir en niveaux de gris
        except ImportError:
            print("PIL non disponible. Utilisez un fichier PGM ou installez Pillow.")
            return None

    # Redimensionner avec letterbox si nécessaire
    if img.size != (TRACES_PER_FRAME, FRAME_HEIGHT):
        print(f"Image originale: {img.size}")
        img = resize_image_with_letterbox(img, TRACES_PER_FRAME, FRAME_HEIGHT)
        print(f"Redimensionnée vers: {img.size} avec bandes noires si nécessaire")

    return np.array(img)

def load_pgm(pgm_path):
    """Charge un fichier PGM"""
    with open(pgm_path, 'rb') as f:
        # Lire l'en-tête PGM - peut être sur une ou plusieurs lignes
        first_line = f.readline().decode().strip()

        if first_line.startswith('P5'):
            parts = first_line.split()
            if len(parts) == 4:  # Format: P5 width height maxval
                _, width, height, maxval = parts
                width, height, maxval = int(width), int(height), int(maxval)
            else:  # Format: P5
                if len(parts) == 1:
                    # Lire dimensions et maxval séparément
                    line = f.readline().decode().strip()
                    while line.startswith('#'):
                        line = f.readline().decode().strip()
                    width, height = map(int, line.split())
                    maxval = int(f.readline().decode().strip())
                else:
                    raise ValueError("Format PGM invalide")
        else:
            raise ValueError("Format PGM P5 requis")

        # Données de l'image
        data = f.read()
        image = np.frombuffer(data, dtype=np.uint8)
        image = image.reshape((height, width))

        return image

def main():
    if len(sys.argv) < 2:
        print("Usage: python encode_image.py <image_file> [output.wav]")
        print("Formats supportés: PGM, PNG (avec PIL)")
        sys.exit(1)

    input_image = sys.argv[1]
    output_wav = sys.argv[2] if len(sys.argv) > 2 else "encoded_image.wav"

    if not os.path.exists(input_image):
        print(f"Erreur: fichier {input_image} introuvable")
        sys.exit(1)

    print(f"Chargement de l'image {input_image}...")
    image_data = load_image(input_image)

    if image_data is None:
        print("Erreur lors du chargement de l'image")
        sys.exit(1)

    print(f"Image chargée: {image_data.shape}")

    # Vérifier les dimensions
    if image_data.shape != (FRAME_HEIGHT, TRACES_PER_FRAME):
        # Transposer si nécessaire
        if image_data.shape == (TRACES_PER_FRAME, FRAME_HEIGHT):
            print("Transposition de l'image pour correspondre au format Voyager")
            image_data = image_data.T
        else:
            print(f"Erreur: dimensions incorrectes {image_data.shape}, attendu {TRACES_PER_FRAME}x{FRAME_HEIGHT}")
            sys.exit(1)

    print("Encodage en signal audio...")
    audio_data = encode_image_to_audio(image_data)

    print(f"Sauvegarde dans {output_wav}...")
    # Normaliser et convertir en int16
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_data * 32767).astype(np.int16)

    wavfile.write(output_wav, SAMPLE_RATE, audio_int16)

    print(f"Fichier WAV créé: {output_wav}")
    print(f"Durée: {len(audio_data)/SAMPLE_RATE:.2f}s")

if __name__ == "__main__":
    main()