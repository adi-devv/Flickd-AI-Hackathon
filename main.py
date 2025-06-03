import cv2
import os
from ultralytics import YOLO
print("YOLOv8 imported successfully")

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    count = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if count % 20 == 0:  # Save every 20th frame
            cv2.imwrite(f"{output_dir}/frame_{count}.jpg", frame)
        count += 1
    vid.release()
    print(f"Extracted Frames to {output_dir}")

# Replace with your sample video path
extract_frames("data/sample_video.mp4", "frames")



def test_yolo(image_path):
    model = YOLO("yolov8n.pt")  # Nano model for speed
    results = model(image_path)
    for result in results:
        print(result.boxes.data)  # Class, bounding box, confidence
    result.save("frames/output.jpg")  # Save annotated image

test_yolo("data/sample_image.jpg")