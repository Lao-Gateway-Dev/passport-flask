import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO("ppmrz11x.pt")  # Replace with your trained model

# Define class names
CLASS_NAMES = {0: "mrz", 1: "passport"}
CONFIDENCE_THRESHOLD = 0.7

# Input image path
image_name = "example.jpg"  # Change this to your image filename
input_folder = "test3"
image_path = os.path.join(input_folder, image_name)

# Read the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to read {image_name}")
else:
    # Run YOLO inference
    results = model(image)

    # Extract detections
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Store passport and MRZ detections
    passports = []
    mrz_regions = []

    for i in range(len(class_ids)):
        if confidences[i] >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = detections[i]
            class_id = class_ids[i]
            confidence = confidences[i]

            if CLASS_NAMES[class_id] == "passport":
                passports.append((x1, y1, x2, y2, confidence))
            elif CLASS_NAMES[class_id] == "mrz":
                mrz_regions.append((x1, y1, x2, y2, confidence))

    # Filter MRZ: Keep only if inside a passport
    filtered_mrz = []
    for mx1, my1, mx2, my2, m_conf in mrz_regions:
        for px1, py1, px2, py2, p_conf in passports:
            if px1 <= mx1 and py1 <= my1 and px2 >= mx2 and py2 >= my2:
                filtered_mrz.append((mx1, my1, mx2, my2, m_conf))
                break  # Ensure one MRZ per passport

    # If no MRZ is inside a passport, skip processing
    if not filtered_mrz:
        print(f"No MRZ detected inside a passport in {image_name}.")
    else:
        # Draw bounding boxes
        for (x1, y1, x2, y2, conf) in passports:
            label = f"Passport {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for (x1, y1, x2, y2, conf) in filtered_mrz:
            label = f"MRZ {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(label)

        # Display image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
