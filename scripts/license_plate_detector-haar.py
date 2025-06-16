import cv2
import numpy as np
import imutils
import argparse
import os

class LicensePlateDetector:
    def __init__(self):
        # Initialize the license plate detector with Haar cascade
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    
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
    
    def draw_detection(self, image, bbox):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

    detector = LicensePlateDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect license plate
        bbox = detector.detect_plate(frame)
        
        # Draw detection
        frame = detector.draw_detection(frame, bbox)
        
        # Write frame if output path is provided
        if writer is not None:
            writer.write(frame)
        
        # Display the frame
        cv2.imshow('License Plate Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
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