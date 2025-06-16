import cv2
import numpy as np
import imutils
import argparse
import os
import easyocr
import time
from datetime import datetime

class LicensePlateDetector:
    def __init__(self):
        # Initialize the license plate detector with Haar cascade
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def detect_plate(self, image):
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try Haar Cascade for license plate detection
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                                    minSize=(25, 25))
        
        # If plates were detected with the cascade classifier
        if len(plates) > 0:
            # Return the first detected plate (x, y, w, h)
            return tuple(plates[0])
        
        # If Haar cascade fails, use edge detection and contour analysis
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        
        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        
        # Loop over contours to find license plate
        for c in cnts:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            
            # If our approximated contour has four points (rectangle), it's likely the license plate
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # Apply basic aspect ratio check (license plates are typically wider than tall)
                aspect_ratio = w / float(h)
                if 1.5 < aspect_ratio < 5.5:
                    return (x, y, w, h)
        
        # If no license plate is detected, return a default region in the center
        h, w = image.shape[:2]
        return (w//4, h//3, w//2, h//4)
    
    def extract_text(self, image, bbox):
        x, y, w, h = bbox
        
        # Extract the license plate region
        plate_img = image[y:y+h, x:x+w]
        
        # Check if plate image is valid
        if plate_img.size == 0:
            return ""
        
        # Preprocess the plate image
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to make text clearer
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use EasyOCR to extract text
        try:
            results = self.reader.readtext(thresh)
            texts = []
            for (_, text, prob) in results:
                # Only include results with confidence above 0.5
                if prob > 0.5:
                    # Clean up the text (remove spaces, special characters)
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    texts.append(cleaned_text)
            
            return ' '.join(texts)
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def draw_detection(self, image, bbox, text=""):
        x, y, w, h = bbox
        # Draw rectangle around license plate
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw text if available
        if text:
            # Create a black background for text
            cv2.rectangle(image, (x, y - 30), (x + w, y), (0, 0, 0), -1)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image

def process_video(input_path, output_path=None):
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create images directory if it doesn't exist
    images_dir = 'images'
    os.makedirs(images_dir, exist_ok=True)

    detector = LicensePlateDetector()
    frame_count = 0
    last_ocr_time = time.time()
    current_text = ""
    last_save_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect license plate
        bbox = detector.detect_plate(frame)
        
        # Perform OCR every 10 frames to save processing time
        if frame_count % 10 == 0 or time.time() - last_ocr_time > 1.0:
            text = detector.extract_text(frame, bbox)
            if text:
                current_text = text
                last_ocr_time = time.time()
        
        # Draw detection with text
        frame = detector.draw_detection(frame, bbox, current_text)
        
        # Write frame if output path is provided
        if writer is not None:
            writer.write(frame)
        
        # Save image every second
        if time.time() - last_save_time > 1.0:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_text = current_text.replace(' ', '_') if current_text else 'unknown'
            filename = f"{images_dir}/plate_{timestamp}_{plate_text}.jpg"
            cv2.imwrite(filename, frame)
            
            # Save just the plate region too
            x, y, w, h = bbox
            plate_img = frame[y:y+h, x:x+w]
            if plate_img.size > 0:
                plate_filename = f"{images_dir}/plate_crop_{timestamp}_{plate_text}.jpg"
                cv2.imwrite(plate_filename, plate_img)
                
            last_save_time = time.time()
            print(f"Saved images: {filename} and plate crop")
        
        frame_count += 1
    
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='License Plate Detection from Video')
    parser.add_argument('--input', '-i', default='videos/car_video.mp4', 
                        help='Path to input video file (default: videos/car_video.mp4)')
    parser.add_argument('--output', '-o', default='output/detected_plates.mp4',
                        help='Path to output video file (default: output/detected_plates.mp4)')
    
    args = parser.parse_args()
    
    # Process the video
    process_video(args.input, args.output)

if __name__ == "__main__":
    main()