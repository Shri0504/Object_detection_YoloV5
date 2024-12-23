import os
import cv2
import torch
from django.shortcuts import render
from collections import OrderedDict

# Load YOLOv5 model
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# View for rendering the home page
def index(request):
    return render(request, 'detection/index.html')

# View for object detection
def detect_objects(request):
    camera = cv2.VideoCapture(0)  # Open the webcam
    detection_results = []  # To store results for display

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture an image.")
            break

        # Perform object detection
        results = yolov5_model(frame)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        detection_results.extend(detections)

        # Render bounding boxes on the frame
        output_frame = results.render()[0]

        cv2.imshow("Object Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
            break

    camera.release()
    cv2.destroyAllWindows()

    # Render the results in the template
    return render(request, 'detection/results.html', {'detections': detection_results})
