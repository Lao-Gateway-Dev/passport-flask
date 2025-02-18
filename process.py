import os
import cv2
import numpy as np
import argparse

def preprocess_mrz(image, image_path, output_path):
    # Create a directory to save processed images if it doesn't exist
    # output_dir = os.path.join(os.path.dirname(image_path), "processed_images")
    os.makedirs(output_path, exist_ok=True)

    # Get the base filename without extension
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=2)
    
    # Save final thresholded image
    output_path = os.path.join(output_path, f"{base_filename}_processed.png")
    success = cv2.imwrite(output_path, thresh)

    if success:
        print(f"[INFO] Processed image saved at: {output_path}")
    else:
        print(f"[ERROR] Failed to save image at: {output_path}")
    return output_path

    # Display the processed image
    # cv2.imshow("Processed Image", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def prepare_image(image_path, output_path):
    print(f"[INFO] Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    output_path = preprocess_mrz(image, image_path, output_path)
    return output_path
