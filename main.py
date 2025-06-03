import cv2
import os
from ultralytics import YOLO
import json

def extract_frames(video_path, output_dir):
    print(f"Attempting to open video: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    os.makedirs(output_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not open video file")
        return
    count = 0
    frames_saved = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if count % 20 == 0:
            frame_path = f"{output_dir}/frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
            frames_saved += 1
            print(f"Saved frame {count} to {frame_path}")
        count += 1
    vid.release()
    print(f"Total frames processed: {count}")
    print(f"Total frames saved: {frames_saved}")

def detect_objects(image_path, frame_number):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return []
    try:
        model = YOLO("yolov8n.pt")
        results = model(image_path)
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                bbox = box.xywh[0].tolist()
                confidence = float(box.conf)
                detections.append({
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": confidence,
                    "frame_number": frame_number
                })
        result.save(f"frames/detected_frame_{frame_number}.jpg")
        return detections
    except Exception as e:
        print(f"Error running YOLOv8: {e}")
        return []

# Process video and frames
video_path = "data/sample_video.mp4"
output_dir = "frames"
extract_frames(video_path, output_dir)
all_detections = []
for frame_path in os.listdir(output_dir):
    if frame_path.endswith(".jpg"):
        frame_number = int(frame_path.split("_")[1].split(".")[0])
        detections = detect_objects(f"{output_dir}/{frame_path}", frame_number)
        all_detections.extend(detections)
with open("frames/detections.json", "w") as f:
    json.dump(all_detections, f, indent=2)
print(f"Saved detections to frames/detections.json")