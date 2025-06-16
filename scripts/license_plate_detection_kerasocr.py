# Step-1 First verify the given keras model is working fine 

# Install required packages
# %pip install matplotlib tensorflow

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext, basename
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import glob
import os
import time
import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Model Loaded successfully...")
        print("Ready to detect license plates...")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_image(img, resize=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_plate(vehicle_img, Dmax=608, Dmin=608):
    ratio = float(max(vehicle_img.shape[:2])) / min(vehicle_img.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(wpod_net, vehicle_img, bound_dim, lp_threshold=0.5)
    return LpImg, cor


def process_video(video_path, output_dir="images"):
    # Create output directories if they don't exist
    plates_dir = os.path.join(output_dir, "plates")
    frames_dir = os.path.join(output_dir, "visualization")
    
    for directory in [plates_dir, frames_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video with {frame_count} frames at {fps} FPS")
    
    frame_idx = 0
    plates_found = 0
    
    # Process frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        # Process every 5th frame to speed things up (adjust as needed)
        if frame_idx % 15 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
            
            # Preprocess the frame
            processed_frame = preprocess_image(frame)
            
            try:
                # Detect license plate
                LpImg, cor = get_plate(processed_frame)
                
                if len(LpImg) > 0:
                    plates_found += 1
                    print(f"License plate detected in frame {frame_idx}")
                    
                    # Save the detected plate
                    plate_img = LpImg[0] * 255  # Convert back to 0-255 scale
                    plate_img = plate_img.astype(np.uint8)
                    plate_path = f"{plates_dir}/plate_frame_{frame_idx}.jpg"
                    cv2.imwrite(plate_path, cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR))
                    
                    # Use keras_ocr to recognize text on the license plate
                    try:
                        # keras_ocr expects RGB images
                        plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                        
                        # Run OCR on the plate image
                        predictions = pipeline.recognize([plate_img_rgb])
                        
                        # Extract the recognized text
                        plate_text = ""
                        if predictions and predictions[0]:
                            # Predictions is a list of (text, box) tuples for each image
                            # Sort by horizontal position (useful for license plates)
                            sorted_predictions = sorted(predictions[0], key=lambda x: x[1][0][0])
                            plate_text = " ".join([word[0] for word in sorted_predictions])
                            # Post-processing for license plates
                            plate_text = plate_text.replace('I', '1')
                            # Convert plate text to uppercase
                            plate_text = plate_text.upper()
                            print(f"Recognized license plate text: {plate_text}")
                        else:
                            print("No text detected on license plate")

                        # Save visualization with OCR results
                        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                        
                        axs[0].imshow(processed_frame)
                        axs[0].set_title(f"Frame {frame_idx}")
                        axs[0].axis('off')
                        
                        axs[1].imshow(LpImg[0])
                        axs[1].set_title("Detected License Plate")
                        axs[1].axis('off')
                        
                        # Visualize OCR results
                        axs[2].imshow(plate_img_rgb)
                        axs[2].set_title(f"OCR Result: {plate_text}")
                        # Draw the bounding boxes for detected text
                        if predictions and predictions[0]:
                            for text, box in predictions[0]:
                                x_coords = [point[0] for point in box]
                                y_coords = [point[1] for point in box]
                                axs[2].plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'r')
                                axs[2].text(box[0][0], box[0][1], text, color='blue', fontsize=12)
                        axs[2].axis('off')
                        
                        plt.tight_layout()
                        
                        # Save the visualization
                        visualization_path = f"{frames_dir}/detection_frame_{frame_idx}.jpg"
                        plt.savefig(visualization_path)
                        plt.close(fig)
                        
                        # Save the recognized text to a file
                        with open(f"{plates_dir}/plate_text_{frame_idx}.txt", "w") as f:
                            f.write(plate_text)
                        
                        print(f"Saved visualization to {visualization_path}")
                        
                    except Exception as e:
                        print(f"Error in OCR processing: {e}")

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    video.release()
    print(f"Video processing complete. Found {plates_found} license plates.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Load the model
    wpod_net_path = "models/wpod-net.json"
    wpod_net = load_model(wpod_net_path)
    
    if wpod_net is not None:
        # Process video file - replace with your video path
        video_path = "/home/ubuntu/Videos/alprVideo.mp4"
        process_video(video_path)