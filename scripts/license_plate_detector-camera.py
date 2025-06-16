import cv2
import numpy as np
import imutils
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

class LicensePlateDetector:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        # Load pre-trained MobileNetV2 as base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom layers for license plate detection
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(4, activation='sigmoid')(x)  # 4 values for bounding box (x, y, w, h)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def preprocess_image(self, image):
        # Resize image to model input size
        resized = cv2.resize(image, (224, 224))
        # Convert to RGB (from BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb.astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def detect_plate(self, image):
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        prediction = self.model.predict(processed_image)[0]
        
        # Convert prediction to bounding box coordinates
        h, w = image.shape[:2]
        x = int(prediction[0] * w)
        y = int(prediction[1] * h)
        width = int(prediction[2] * w)
        height = int(prediction[3] * h)
        
        return (x, y, width, height)
    
    def draw_detection(self, image, bbox):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

def main():
    # Initialize video capture (0 for webcam)
    cap = cv2.VideoCapture(0)
    detector = LicensePlateDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect license plate
        bbox = detector.detect_plate(frame)
        
        # Draw detection
        frame = detector.draw_detection(frame, bbox)
        
        # Display the frame
        cv2.imshow('License Plate Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 