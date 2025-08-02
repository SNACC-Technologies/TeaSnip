# TeaSnip: Intelligent Image Cropping Tool

TeaSnip is a computer vision tool that detects persons in images and crops them while maintaining either:
- The original image aspect ratio (default)
- A specified aspect ratio (via command line option)

## Features

- Uses YOLOv8 for accurate person detection
- Preserves original aspect ratio by default
- Supports custom aspect ratios when needed
- Ensures entire person is visible in output
- Processes single images or entire directories
- Command-line interface with configurable options
- Supports JPEG and PNG input formats
- Outputs high-quality PNG images
- Adjustable confidence threshold for person detection
- Verbose logging for troubleshooting
- Handles various edge cases gracefully

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install them manually:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

## Usage

```bash
python teasnip.py <input_image_path|input_directory> [output_image_path|output_directory] [options]
```

### Examples

```bash
# Process a single image and save with "_cropped" suffix
python teasnip.py input.jpg

# Process a single image and save to specific output path
python teasnip.py input.jpg output.png

# Process all images in a directory
python teasnip.py input_directory/ output_directory/

# Process with custom confidence threshold
python teasnip.py input.jpg -c 0.7

# Process with verbose logging
python teasnip.py input.jpg -v

# Process with custom model
python teasnip.py input.jpg -m yolov8s.pt

# Process with custom aspect ratio (4:5)
python teasnip.py input.jpg -a 4:5

# Process with input image aspect ratio (no -a flag)
python teasnip.py input.jpg
```

### Options

- `-c, --confidence`: Confidence threshold for person detection (default: 0.5)
- `-m, --model`: Path to YOLO model file (default: yolov8n.pt)
- `-a, --aspect-ratio`: Output aspect ratio as W:H (default: use input image ratio)
- `-v, --verbose`: Enable verbose logging

## How It Works

1. Loads the input image
2. Uses YOLOv8 to detect persons in the image (class 0 in COCO dataset)
3. Selects the person with highest confidence if multiple persons are detected
4. Calculates optimal crop dimensions to maintain either:
   - The original image aspect ratio (default)
   - A specified aspect ratio (if provided)
5. Ensures the entire person is visible within the cropped image
6. Saves the resulting cropped image as a PNG file

## Requirements

- Python 3.6+
- ultralytics >= 8.0.0
- opencv-python >= 4.0.0
- numpy >= 1.20.0

## Notes

- The tool processes both JPEG (.jpg, .jpeg) and PNG (.png) images
- If no person is detected, the tool will return an error
- The first time you run the tool, it will download the YOLOv8 model (yolov8n.pt)
- Output images are always saved as PNG files for best quality
- For batch processing, input and output directories are specified as arguments

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
