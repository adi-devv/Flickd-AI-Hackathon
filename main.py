import cv2
import os

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    count = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if count % 20 == 0:  # Save every 10th frame
            cv2.imwrite(f"{output_dir}/frame_{count}.jpg", frame)
        count += 1
    vid.release()
    print(f"Extracted Frames to {output_dir}")

# Replace with your sample video path
extract_frames("data/sample_video.mp4", "frames")