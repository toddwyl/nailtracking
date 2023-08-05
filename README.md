# Paint your nails with TensorFlow & OpenCV

This Python project is a handnail tracking application using TensorFlow and OpenCV. It uses a pre-trained object detection model to detect handnails in real-time from the camera (Webcam) video stream.

<!-- ![Demo](demo.gif) -->

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Tkinter (for interface use)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

2. Install the dependencies:

```bash
pip install tensorflow opencv-python numpy
```

## Usage

Run the `handnail_tracking.py` script to start the handnail tracking application:

```bash
python handnail_tracking.py
```

The application will begin capturing video from the camera and display the video frames with detected handnails in real-time. To exit the application, press the "q" key.

## Customization

If desired, you can adjust the parameters in the handnail_tracking.py file to suit your needs, such as the pre-trained model, colors used for the ellipses, and opacity.

## Credits

- The object detection model is based on the TensorFlow Object Detection API.

- Handnail detection is inspired by [Wen YongLiang's Nail Tracking algorythm](https://github.com/toddwyl/nailtracking).
