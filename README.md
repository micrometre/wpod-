# License Plate Detection Application

This application uses computer vision and deep learning to detect license plates in real-time using a webcam feed.

## Features

- Real-time license plate detection
- Uses MobileNetV2 as the base model for efficient detection
- Works with both images and video streams
- Simple and intuitive interface

## Requirements

- Python 3.8 or higher
- Webcam or video input device
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd object-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python license_plate_detector.py
```

2. The application will open your webcam feed and start detecting license plates in real-time.
3. Press 'q' to quit the application.

## Model Training

Note: The current implementation uses a pre-trained model. To train the model on your own dataset:

1. Prepare your dataset of license plate images
2. Modify the training parameters in the code
3. Run the training script

## License

This project is licensed under the MIT License - see the LICENSE file for details. 