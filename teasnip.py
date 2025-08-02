#!/usr/bin/env python3
"""
TeaSnip: Social Media Portrait Cropping Tool

This script processes images to detect a person and crop the image to maintain 
a 9:16 aspect ratio (portrait orientation) perfect for social media platforms.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import math
import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonCropper:
    """A class to handle person detection and cropping with 9:16 aspect ratio."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', aspect_ratio: Optional[str] = None):
        """
        Initialize the PersonCropper with a YOLO model and optional aspect ratio.
        
        Args:
            model_path (str): Path to the YOLO model file
            aspect_ratio (str, optional): Desired aspect ratio as 'width:height'.
                If None, uses input image ratio.
        """
        self.model = YOLO(model_path)
        self.target_aspect_ratio = None
        if aspect_ratio:
            try:
                width, height = map(float, aspect_ratio.split(':'))
                self.target_aspect_ratio = width / height
            except (ValueError, AttributeError):
                logger.warning(f"Invalid aspect ratio '{aspect_ratio}', will use input image ratio")
    
    def detect_persons(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[np.ndarray, float]]:
        """
        Detect persons in an image using YOLOv8.
        
        Args:
            image (np.ndarray): Input image
            confidence_threshold (float): Minimum confidence for detection
            
        Returns:
            List[Tuple[np.ndarray, float]]: List of (bounding_box, confidence) tuples
        """
        # Run YOLOv8 detection with specific class filtering for person (class 0)
        results = self.model(image, classes=[0], conf=confidence_threshold)
        
        # Find person detections with confidence threshold
        person_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Class 0 in COCO dataset is 'person'
                    if cls == 0 and conf >= confidence_threshold:
                        person_detections.append((box.xyxy[0].cpu().numpy(), conf))
        
        return person_detections
    
    def calculate_optimal_crop(self, image_shape: Tuple[int, int], person_box: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate optimal crop coordinates for 9:16 aspect ratio.
        
        Args:
            image_shape (Tuple[int, int]): Shape of the input image (height, width)
            person_box (np.ndarray): Person bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple[int, int, int, int]: Crop coordinates (x1, y1, x2, y2)
        """
        height, width = image_shape[:2]
        x1, y1, x2, y2 = person_box.astype(int)
        
        # Calculate person dimensions
        person_width = x2 - x1
        person_height = y2 - y1
        
        # Calculate minimum crop dimensions to fit the person
        # For 9:16 aspect ratio, if we know the height, width = height * (9/16)
        # If we know the width, height = width / (9/16)
        
        # Option 1: Base crop height on person height
        crop_height_option1 = person_height
        crop_width_option1 = int(crop_height_option1 * self.target_aspect_ratio)
        
        # Option 2: Base crop width on person width
        crop_width_option2 = person_width
        crop_height_option2 = int(crop_width_option2 / self.target_aspect_ratio)
        
        # Choose the option that ensures the entire person fits
        if crop_width_option1 >= person_width:
            # Option 1 works - person width fits in crop width
            crop_width = crop_width_option1
            crop_height = crop_height_option1
        elif crop_height_option2 >= person_height:
            # Option 2 works - person height fits in crop height
            crop_width = crop_width_option2
            crop_height = crop_height_option2
        else:
            # Neither option works perfectly, choose the one that fits better
            crop_width = max(crop_width_option2, person_width)
            crop_height = crop_height_option2
        
        # Ensure crop dimensions don't exceed image dimensions
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)
        
        # Adjust aspect ratio to exactly 9:16 if possible
        current_ratio = crop_width / crop_height
        if current_ratio > self.target_aspect_ratio:
            # Width is too large, adjust it
            crop_width = int(crop_height * self.target_aspect_ratio)
        elif current_ratio < self.target_aspect_ratio:
            # Height is too large, adjust it
            crop_height = int(crop_width / self.target_aspect_ratio)
        
        # Calculate crop coordinates to center the person
        crop_center_x = (x1 + x2) // 2
        crop_center_y = (y1 + y2) // 2
        
        crop_x1 = crop_center_x - crop_width // 2
        crop_y1 = crop_center_y - crop_height // 2
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height
        
        # Adjust crop to ensure it's within image boundaries
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > width:
            crop_x1 -= (crop_x2 - width)
            crop_x2 = width
        if crop_y2 > height:
            crop_y1 -= (crop_y2 - height)
            crop_y2 = height
        
        # Final adjustment to ensure we still have the correct aspect ratio
        # and the person is fully visible
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        
        # Ensure the person is fully within the crop
        if x1 < crop_x1:
            # Person extends left of crop, move crop left
            offset = crop_x1 - x1
            crop_x1 = x1
            crop_x2 = min(width, crop_x2 - offset)
        if y1 < crop_y1:
            # Person extends above crop, move crop up
            offset = crop_y1 - y1
            crop_y1 = y1
            crop_y2 = min(height, crop_y2 - offset)
        if x2 > crop_x2:
            # Person extends right of crop, move crop right
            offset = x2 - crop_x2
            crop_x2 = x2
            crop_x1 = max(0, crop_x1 + offset)
        if y2 > crop_y2:
            # Person extends below crop, move crop down
            offset = y2 - crop_y2
            crop_y2 = y2
            crop_y1 = max(0, crop_y1 + offset)
        
        # Final adjustment for exact 9:16 aspect ratio with standard resolutions
        final_width = crop_x2 - crop_x1
        final_height = crop_y2 - crop_y1
        
        # Round up to nearest standard resolution (multiples of 9:16)
        min_n_width = math.ceil(final_width / 9)
        min_n_height = math.ceil(final_height / 16)
        n = max(min_n_width, min_n_height)
        
        # Use the smallest possible standard resolution that fits the person
        standard_width = n * 9
        standard_height = n * 16
        
        # Make sure the standard resolution doesn't exceed image dimensions
        standard_width = min(standard_width, width)
        standard_height = min(standard_height, height)
        
        # Adjust to maintain exact 9:16 aspect ratio
        aspect_ratio = standard_width / standard_height
        
        if aspect_ratio > self.target_aspect_ratio:
            # Width is too large, adjust it
            standard_width = int(standard_height * self.target_aspect_ratio)
        elif aspect_ratio < self.target_aspect_ratio:
            # Height is too large, adjust it
            standard_height = int(standard_width / self.target_aspect_ratio)
        
        # Center the crop around the person center
        center_x = (crop_x1 + crop_x2) // 2
        center_y = (crop_y1 + crop_y2) // 2
        
        # Calculate new crop coordinates with standard dimensions
        new_crop_x1 = max(0, center_x - standard_width // 2)
        new_crop_y1 = max(0, center_y - standard_height // 2)
        new_crop_x2 = min(width, new_crop_x1 + standard_width)
        new_crop_y2 = min(height, new_crop_y1 + standard_height)
        
        # Adjust if crop exceeds image boundaries
        if new_crop_x2 - new_crop_x1 < standard_width:
            new_crop_x1 = max(0, new_crop_x2 - standard_width)
        if new_crop_y2 - new_crop_y1 < standard_height:
            new_crop_y1 = max(0, new_crop_y2 - standard_height)
        
        # Ensure exact 9:16 aspect ratio
        final_crop_width = new_crop_x2 - new_crop_x1
        final_crop_height = new_crop_y2 - new_crop_y1
        
        if final_crop_width / final_crop_height > self.target_aspect_ratio:
            # Width is too large, reduce it
            adjusted_width = int(final_crop_height * self.target_aspect_ratio)
            # Center horizontally
            center_x = (new_crop_x1 + new_crop_x2) // 2
            new_crop_x1 = max(0, center_x - adjusted_width // 2)
            new_crop_x2 = min(width, new_crop_x1 + adjusted_width)
        elif final_crop_width / final_crop_height < self.target_aspect_ratio:
            # Height is too large, reduce it
            adjusted_height = int(final_crop_width / self.target_aspect_ratio)
            # Center vertically
            center_y = (new_crop_y1 + new_crop_y2) // 2
            new_crop_y1 = max(0, center_y - adjusted_height // 2)
            new_crop_y2 = min(height, new_crop_y1 + adjusted_height)
        
        return new_crop_x1, new_crop_y1, new_crop_x2, new_crop_y2
    
    def crop_person(self, image_path: str, output_path: Optional[str] = None,
                   confidence_threshold: float = 0.5) -> bool:
        """
        Crop an image to specified aspect ratio (or input image ratio) while keeping the entire person visible.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path for output image (optional)
            confidence_threshold (float): Minimum confidence for person detection
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image {image_path}")
                return False
            
            height, width = image.shape[:2]
            logger.info(f"Processing image of size {width}x{height}")
            
            # Detect persons in the image
            person_detections = self.detect_persons(image, confidence_threshold)
            
            if not person_detections:
                logger.error("No person detected in the image")
                return False
            
            # Use the person with highest confidence
            best_person_idx = np.argmax([conf for _, conf in person_detections])
            person_box, person_conf = person_detections[best_person_idx]
            logger.info(f"Detected person with confidence {person_conf:.2f}")
            
            # Calculate optimal crop
            # Use input image ratio if no aspect ratio specified
            if self.target_aspect_ratio is None:
                img_height, img_width = image.shape[:2]
                self.target_aspect_ratio = img_width / img_height
                logger.info(f"Using input image aspect ratio: {img_width}:{img_height}")
                
            crop_coords = self.calculate_optimal_crop(image.shape, person_box)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
            
            # Apply crop
            cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Save output image as PNG
            if output_path is None:
                input_path = Path(image_path)
                output_path = input_path.parent / f"{input_path.stem}_cropped.png"
            else:
                # Ensure output is PNG
                output_path_obj = Path(output_path)
                if output_path_obj.suffix.lower() not in ['.png']:
                    output_path = output_path_obj.with_suffix('.png')
            
            success = cv2.imwrite(str(output_path), cropped_image)
            
            if success:
                final_width = crop_x2 - crop_x1
                final_height = crop_y2 - crop_y1
                aspect_ratio = final_height / final_width
                logger.info(f"Successfully cropped image saved to {output_path}")
                target_ratio = "input image ratio" if self.target_aspect_ratio is None else "9:16"
                logger.info(f"Cropped to {final_width}x{final_height} pixels with {target_ratio} aspect ratio")
                logger.info(f"Aspect ratio (height/width): {aspect_ratio:.6f}")
                return True
            else:
                logger.error(f"Error saving cropped image to {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return False

def process_single_image(cropper: PersonCropper, input_path: str, output_path: Optional[str] = None, 
                        confidence_threshold: float = 0.5) -> bool:
    """Process a single image file"""
    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist")
        return False
    
    # Check if input is a JPEG file
    if not input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        logger.warning(f"Input file {input_path} is not a standard image format. Attempting to process anyway.")
    
    return cropper.crop_person_to_9_16_aspect(input_path, output_path, confidence_threshold)

def process_directory(cropper: PersonCropper, input_dir: str, output_dir: str, 
                     confidence_threshold: float = 0.5) -> bool:
    """Process all image files in a directory"""
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        logger.warning(f"No image files found in directory {input_dir}")
        return False
    
    logger.info(f"Processing {len(image_files)} image files from {input_dir} to {output_dir}")
    
    success_count = 0
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"{Path(image_file).stem}_cropped.png")
        
        logger.info(f"Processing {image_file}...")
        if process_single_image(cropper, input_path, output_path, confidence_threshold):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(image_files)} images")
    return success_count > 0

def main():
    """Main function to handle command line arguments and execution"""
    parser = argparse.ArgumentParser(
        description="TeaSnip: Crop images to 9:16 aspect ratio for social media",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python teasnip.py input.jpg
  python teasnip.py input.jpg output.png
  python teasnip.py input_dir/ output_dir/
  python teasnip.py input.jpg -c 0.7
        """
    )
    
    parser.add_argument("input_path", help="Input image file or directory")
    parser.add_argument("output_path", nargs="?", help="Output image file or directory (optional)")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, 
                        help="Confidence threshold for person detection (default: 0.5)")
    parser.add_argument("-m", "--model", default="yolov8n.pt", 
                        help="Path to YOLO model file (default: yolov8n.pt)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("-a", "--aspect-ratio",
                        help="Desired output aspect ratio as W:H (default: use input image ratio)")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize cropper
    cropper = PersonCropper(args.model, args.aspect_ratio)
    
    # Check if input is a directory
    if os.path.isdir(args.input_path):
        # Batch processing mode
        output_dir = args.output_path if args.output_path else "cropped_output"
        success = process_directory(cropper, args.input_path, output_dir, args.confidence)
        if not success:
            sys.exit(1)
    else:
        # Single image processing mode
        success = process_single_image(cropper, args.input_path, args.output_path, args.confidence)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()