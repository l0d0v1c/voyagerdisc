# Voyager Golden Record Image Decoder

A Python script to extract images encoded as audio on the Voyager Golden Record.

Based on [foodini/voyager](https://github.com/foodini/voyager).

## How the Images are Encoded

The Voyager Golden Record contains 116 images encoded as analog audio signals at **384 kHz** sample rate (78 images per stereo channel).


![](left_021.png)

![](left_076.png)

### Signal Format

Each image is composed of **540 traces** (scan lines), with each trace containing **364 pixels**.

#### Trace Structure
- **Sync pulse**: Each trace begins with a high-amplitude spike followed by a falling edge
- **Image data**: ~2700 samples representing 364 pixels
- **Trace spacing**: Alternates between ~3100 and ~3300 samples

#### Pixel Encoding
- Pixel intensity is encoded as the **average amplitude** of the audio signal
- Lower amplitude = brighter pixel (inverted)
- Signal range: approximately -0.26 to +0.18
- A cosine-based lookup table maps signal values to 0-255 intensity

### Image Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 384,000 Hz |
| Frame height | 364 pixels |
| Traces per frame | 540 |
| Samples per trace | 3000-3400 |
| Images per channel | 78 |
| Total images | 156 (116 unique + color channels) |

### Color Images

Some images are transmitted as three separate frames (Red, Green, Blue channels) that must be combined to produce a color image. The encoding includes metadata arrays specifying:
- Which channel (R/G/B or grayscale) each frame represents
- Image orientation (normal, rotate left, rotate right)

## Requirements

### For Image Extraction
- Python 3
- NumPy
- SciPy
- FFmpeg (for non-WAV input files)

```bash
pip install numpy scipy
```

### For Image Encoding
- All above requirements, plus:
- Pillow (PIL) for image processing and automatic resizing

```bash
pip install numpy scipy pillow
```

## Usage

### Image Extraction (from Voyager Golden Record)

```bash
# From WAV file (384 kHz stereo)
python extract_images.py 384kHzStereo.wav

# From FLAC file
python extract_images.py 384kHzStereo.flac
```

Images are extracted to `extracted_images/` directory in PGM format (grayscale).

### Image Encoding/Decoding

This repository also includes tools to encode your own images into Voyager-compatible audio files:

#### Single Image Encoding

```bash
# Encode single image to WAV (Voyager format)
python encode_image.py image.png output.wav
```

**Features:**
- Automatic resizing to 540×364 pixels
- Letterbox with black bars for different aspect ratios
- Supports PNG, JPG, PGM formats
- Compatible with `extract_images.py`

#### Stereo Image Encoding (Two Images)

```bash
# Encode two images to stereo WAV (simplified format)
python encode_simple_stereo.py left_image.png right_image.png output.wav

# Decode stereo WAV back to two images
python extract_simple_stereo.py input.wav output_directory/
```

**Features:**
- Two images in one stereo file (left/right channels)
- Perfect quality preservation
- Automatic resizing with letterbox
- Direct pixel-to-audio mapping
- Shorter files (~0.5s vs 7s)
- Outputs both PGM and PNG formats

#### Stereo Image Encoding (Voyager-compatible)

```bash
# Encode two images to Voyager-compatible stereo WAV
python encode_stereo_images.py left_image.png right_image.png output.wav

# Decode with Voyager-compatible decoder
python extract_stereo_images.py input.wav output_directory/
```

**Note:** The simple format (`encode_simple_stereo.py`) provides better quality and efficiency for custom image encoding, while the Voyager-compatible format maintains the original signal structure.

### Examples

```bash
# Basic workflow: encode and decode
python encode_simple_stereo.py photo1.jpg photo2.png my_images.wav
python extract_simple_stereo.py my_images.wav decoded/

# Single image encoding
python encode_image.py portrait.png encoded_portrait.wav

# Different input formats supported
python encode_simple_stereo.py image.jpg image.png output.wav
python encode_simple_stereo.py image.pgm image.bmp output.wav
```

### Output Formats

| Script | Input | Output | Duration | Quality |
|--------|-------|--------|----------|---------|
| `encode_image.py` | 1 image | WAV mono ~7s | Long | Voyager-compatible |
| `encode_simple_stereo.py` | 2 images | WAV stereo ~0.5s | Short | Perfect |
| `encode_stereo_images.py` | 2 images | WAV stereo ~7s | Long | Voyager-compatible |

## Audio File Recommendations

- **WAV/FLAC**: Lossless, best quality
- **Opus/MP3**: Lossy compression degrades image quality significantly because:
  - MP3 is limited to 48 kHz sample rate (loses 87% of data)
  - Lossy codecs are optimized for human hearing, not data signals

## References

- [How to Decode the Images on the Voyager Golden Record](https://boingboing.net/2017/09/05/how-to-decode-the-images-on-th.html)
- [Original audio files](https://drive.google.com/drive/folders/0B0Swx_1rwA6XcFFLc29ncFJSZmM)
