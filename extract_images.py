#!/usr/bin/env python3
"""
Extrait les images du Voyager Golden Record depuis un fichier audio MP3.
Les images sont encodées sous forme de signal audio à 384kHz.
"""

import numpy as np
from scipy.io import wavfile
import os
import math

# Paramètres du format Voyager
SAMPLE_RATE = 384000
FRAME_HEIGHT = 364
TRACES_PER_FRAME = 540

# Points de départ pour chaque image (canal gauche et droit)
START_POINTS = [
    # Canal gauche (78 images)
    [
        15*SAMPLE_RATE + 240351, 21*SAMPLE_RATE + 245777, 27*SAMPLE_RATE + 299713,
        33*SAMPLE_RATE + 267909, 39*SAMPLE_RATE + 209551, 45*SAMPLE_RATE + 148078,
        51*SAMPLE_RATE + 26745, 56*SAMPLE_RATE + 211195, 62*SAMPLE_RATE + 13271,
        67*SAMPLE_RATE + 216256, 74*SAMPLE_RATE + 256650, 80*SAMPLE_RATE + 173217,
        86*SAMPLE_RATE + 148759, 92*SAMPLE_RATE + 116906, 97*SAMPLE_RATE + 366248,
        103*SAMPLE_RATE + 269951, 109*SAMPLE_RATE + 116120, 114*SAMPLE_RATE + 269542,
        120*SAMPLE_RATE + 154943, 126*SAMPLE_RATE + 91019, 131*SAMPLE_RATE + 268886,
        137*SAMPLE_RATE + 142217, 143*SAMPLE_RATE + 40894, 149*SAMPLE_RATE + 79996,
        155*SAMPLE_RATE + 4382, 160*SAMPLE_RATE + 265923, 166*SAMPLE_RATE + 148792,
        172*SAMPLE_RATE + 84628, 177*SAMPLE_RATE + 311218, 183*SAMPLE_RATE + 235149,
        189*SAMPLE_RATE + 175178, 194*SAMPLE_RATE + 358166, 201*SAMPLE_RATE + 60411,
        207*SAMPLE_RATE + 16150, 212*SAMPLE_RATE + 308295, 218*SAMPLE_RATE + 236789,
        224*SAMPLE_RATE + 155253, 230*SAMPLE_RATE + 42514, 236*SAMPLE_RATE + 113295,
        241*SAMPLE_RATE + 366821, 247*SAMPLE_RATE + 314987, 253*SAMPLE_RATE + 240089,
        259*SAMPLE_RATE + 173965, 265*SAMPLE_RATE + 121353, 271*SAMPLE_RATE + 70064,
        277*SAMPLE_RATE + 23299, 282*SAMPLE_RATE + 365023, 288*SAMPLE_RATE + 193849,
        294*SAMPLE_RATE + 93673, 300*SAMPLE_RATE + 14155, 305*SAMPLE_RATE + 346213,
        311*SAMPLE_RATE + 307560, 317*SAMPLE_RATE + 147218, 323*SAMPLE_RATE + 27443,
        328*SAMPLE_RATE + 254251, 334*SAMPLE_RATE + 126710, 340*SAMPLE_RATE + 36844,
        345*SAMPLE_RATE + 286322, 351*SAMPLE_RATE + 164075, 357*SAMPLE_RATE + 78195,
        362*SAMPLE_RATE + 263903, 368*SAMPLE_RATE + 149972, 373*SAMPLE_RATE + 367679,
        379*SAMPLE_RATE + 287739, 385*SAMPLE_RATE + 230564, 391*SAMPLE_RATE + 176842,
        397*SAMPLE_RATE + 16062, 402*SAMPLE_RATE + 368972, 408*SAMPLE_RATE + 179635,
        414*SAMPLE_RATE + 159821, 420*SAMPLE_RATE + 158215, 427*SAMPLE_RATE + 51185,
        432*SAMPLE_RATE + 250211, 438*SAMPLE_RATE + 174216, 443*SAMPLE_RATE + 378736,
        449*SAMPLE_RATE + 236946, 455*SAMPLE_RATE + 94877, 461*SAMPLE_RATE + 201612,
    ],
    # Canal droit (78 images)
    [
        16*SAMPLE_RATE + 362529, 22*SAMPLE_RATE + 209325, 28*SAMPLE_RATE + 173272,
        34*SAMPLE_RATE + 164865, 40*SAMPLE_RATE + 98143, 45*SAMPLE_RATE + 376141,
        51*SAMPLE_RATE + 218597, 57*SAMPLE_RATE + 52853, 62*SAMPLE_RATE + 340679,
        68*SAMPLE_RATE + 203243, 74*SAMPLE_RATE + 205555, 80*SAMPLE_RATE + 69735,
        85*SAMPLE_RATE + 346872, 91*SAMPLE_RATE + 280177, 97*SAMPLE_RATE + 172925,
        103*SAMPLE_RATE + 34621, 108*SAMPLE_RATE + 341194, 114*SAMPLE_RATE + 226775,
        120*SAMPLE_RATE + 139779, 126*SAMPLE_RATE + 65050, 131*SAMPLE_RATE + 351292,
        137*SAMPLE_RATE + 227618, 143*SAMPLE_RATE + 146940, 149*SAMPLE_RATE + 88974,
        154*SAMPLE_RATE + 374392, 160*SAMPLE_RATE + 305053, 166*SAMPLE_RATE + 245569,
        172*SAMPLE_RATE + 113060, 177*SAMPLE_RATE + 371073, 183*SAMPLE_RATE + 381342,
        189*SAMPLE_RATE + 253823, 195*SAMPLE_RATE + 200249, 201*SAMPLE_RATE + 98237,
        206*SAMPLE_RATE + 366832, 212*SAMPLE_RATE + 224703, 218*SAMPLE_RATE + 85423,
        223*SAMPLE_RATE + 349160, 229*SAMPLE_RATE + 247995, 235*SAMPLE_RATE + 131829,
        241*SAMPLE_RATE + 81391, 246*SAMPLE_RATE + 336806, 252*SAMPLE_RATE + 213356,
        258*SAMPLE_RATE + 139425, 263*SAMPLE_RATE + 352645, 269*SAMPLE_RATE + 306314,
        275*SAMPLE_RATE + 128031, 281*SAMPLE_RATE + 31592, 286*SAMPLE_RATE + 343591,
        292*SAMPLE_RATE + 281969, 298*SAMPLE_RATE + 232050, 304*SAMPLE_RATE + 166472,
        310*SAMPLE_RATE + 169232, 316*SAMPLE_RATE + 76140, 322*SAMPLE_RATE + 35834,
        328*SAMPLE_RATE + 18093, 333*SAMPLE_RATE + 280296, 339*SAMPLE_RATE + 66769,
        344*SAMPLE_RATE + 363619, 350*SAMPLE_RATE + 281995, 356*SAMPLE_RATE + 189398,
        362*SAMPLE_RATE + 68027, 367*SAMPLE_RATE + 325978, 373*SAMPLE_RATE + 219591,
        379*SAMPLE_RATE + 196994, 385*SAMPLE_RATE + 133714, 391*SAMPLE_RATE + 37879,
        396*SAMPLE_RATE + 331129, 402*SAMPLE_RATE + 293135, 408*SAMPLE_RATE + 250696,
        414*SAMPLE_RATE + 310101, 420*SAMPLE_RATE + 233345, 426*SAMPLE_RATE + 366703,
        432*SAMPLE_RATE + 229409, 438*SAMPLE_RATE + 120560, 444*SAMPLE_RATE + 86063,
        449*SAMPLE_RATE + 319515, 455*SAMPLE_RATE + 187808, 461*SAMPLE_RATE + 60886,
    ],
]


def float_to_intensity(f):
    """Convertit une valeur audio en intensité de pixel"""
    lower_bound = -0.26
    upper_bound = 0.18
    if f <= lower_bound:
        return 255
    if f >= upper_bound:
        return 0
    t = (upper_bound - f) / (upper_bound - lower_bound)
    cos_term = math.cos(t * math.pi)
    return int(255 - (cos_term + 1.0) * 255.0 / 2.0)

def find_trace_start(samples, start_offset):
    """Trouve le début d'une trace en cherchant le pic de sync"""
    end = min(start_offset + 190, len(samples))
    max_idx = start_offset
    max_val = -1.0
    for i in range(start_offset, end):
        if samples[i] > max_val:
            max_val = samples[i]
            max_idx = i

    # Trouve le minimum après le maximum
    end = min(max_idx + 190, len(samples))
    min_idx = max_idx
    min_val = 1.0
    for i in range(max_idx, end):
        if samples[i] < min_val:
            min_val = samples[i]
            min_idx = i

    return min_idx

def decode_image(samples, start_point, output_path):
    """Décode une image depuis les échantillons audio"""
    buffer_size = TRACES_PER_FRAME * 3400 * 3 // 2

    if start_point + buffer_size > len(samples):
        print(f"  Pas assez de données pour {output_path}")
        return False

    buffer = samples[start_point:start_point + buffer_size]
    frame = np.zeros((TRACES_PER_FRAME, FRAME_HEIGHT), dtype=np.uint8)

    trace_start = find_trace_start(buffer, 0)

    for trace_id in range(TRACES_PER_FRAME):
        trace_end = find_trace_start(buffer, trace_start + 3000)

        # Calcul des bornes de l'image
        if trace_id < 164:
            image_start = trace_start + 220
            image_end = trace_start + 2900
        else:
            image_start = trace_start + 220
            image_end = trace_start + 2897

        if trace_id % 2 == 0:
            image_start -= 12
            image_end -= 12

        image_width = image_end - image_start - 1
        pixel_width = image_width / FRAME_HEIGHT
        pixel_start = float(image_start)

        for pixel_id in range(FRAME_HEIGHT):
            pixel_end = pixel_start + pixel_width
            ps, pe = int(pixel_start), int(pixel_end)

            if pe > len(buffer):
                break

            # Moyenne du signal
            avg = np.mean(buffer[ps:pe]) if pe > ps else 0
            frame[trace_id, pixel_id] = float_to_intensity(avg)
            pixel_start = pixel_end

        trace_start = trace_end

    # Sauvegarde en PGM
    with open(output_path, 'wb') as f:
        f.write(f"P5 {TRACES_PER_FRAME} {FRAME_HEIGHT} 255\n".encode())
        for y in range(FRAME_HEIGHT):
            for x in range(TRACES_PER_FRAME):
                f.write(bytes([frame[x, y]]))

    return True

def main():
    import sys
    import subprocess
    import tempfile

    input_file = sys.argv[1] if len(sys.argv) > 1 else "384kHzStereo.wav"
    output_dir = "extracted_images"

    os.makedirs(output_dir, exist_ok=True)

    # Si c'est un FLAC ou autre format, convertir en WAV temporaire
    if not input_file.endswith('.wav'):
        print(f"Conversion de {input_file} en WAV...")
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file,
            '-ar', str(SAMPLE_RATE), temp_wav.name
        ], check=True, capture_output=True)
        wav_file = temp_wav.name
        cleanup_wav = True
    else:
        wav_file = input_file
        cleanup_wav = False

    print(f"Chargement de {wav_file}...")
    sample_rate, audio = wavfile.read(wav_file)
    print(f"Sample rate: {sample_rate} Hz, durée: {len(audio)/sample_rate:.1f}s")

    if cleanup_wav:
        os.remove(wav_file)

    # Normalisation en float [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Extraction des canaux
    if len(audio.shape) > 1:
        left = audio[:, 0]
        right = audio[:, 1]
    else:
        left = right = audio

    channels = [left, right]
    channel_names = ["left", "right"]

    # Décodage des images
    for ch in range(2):
        print(f"\nDécodage canal {channel_names[ch]}...")
        for i, start in enumerate(START_POINTS[ch]):
            output = os.path.join(output_dir, f"{channel_names[ch]}_{i:03d}.pgm")
            if decode_image(channels[ch], start, output):
                print(f"  Image {i:03d} OK")

    print(f"\nImages extraites dans {output_dir}/")

if __name__ == "__main__":
    main()
