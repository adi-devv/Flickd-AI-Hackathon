import cv2
import os
from ultralytics import YOLO
import json
import clip
import torch
import pandas as pd
import faiss
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import spacy

def extract_frames(video_path, output_dir):
    print(f"Attempting to open video: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    os.makedirs(output_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not open video file")
        vid.release()
        return False
    count = 0
    frames_saved = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if count % 20 == 0:
            frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_saved += 1
            print(f"Saved frame {count} to {frame_path}")
        count += 1
    vid.release()
    print(f"Total frames processed: {count}")
    print(f"Total frames saved: {frames_saved}")
    return True

def detect_objects(image_path, frame_number, model, detectedframepath):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return []
    try:
        frame = cv2.imread(image_path)
        results = model.predict(source=image_path, conf=0.4, save=False, line_width=2)
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                bbox = box.xywh[0].tolist()
                confidence = float(box.conf)
                x, y, w, h = [int(v) for v in bbox]
                crop = frame[y:y+h, x:x+w]
                detections.append({
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": confidence,
                    "frame_number": frame_number,
                    "crop": crop
                })
            annotated_frame = result.plot()
            save_path = os.path.join(detectedframepath, f"detected_frame_{frame_number}.jpg")
            cv2.imwrite(save_path, annotated_frame)
            print(f"Saved detected frame to {save_path}")
        return detections
    except Exception as e:
        print(f"Error running YOLO: {e}")
        return []

def setup_faiss_index(images_csv, product_data_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    images_df = pd.read_csv(images_csv)
    product_df = pd.read_csv(product_data_csv)
    catalog = images_df.merge(product_df, on="id", how="inner")
    
    embeddings = []
    product_info = []
    product_id_to_indices = {}
    current_index = 0
    
    for product_id, group in catalog.groupby('id'):
        product_indices = []
        product_data = group.iloc[0]
        for _, row in group.iterrows():
            try:
                response = requests.get(row['image_url'], timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                image = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = clip_model.encode_image(image).cpu().numpy()
                embeddings.append(embedding)
                product_indices.append(current_index)
                current_index += 1
            except Exception as e:
                print(f"Error processing catalog image {row['image_url']}: {e}")
        product_info.append({
            "id": product_id,
            "product_type": product_data['product_type'],
            "description": product_data['description'],
            "product_tags": product_data['product_tags']
        })
        product_id_to_indices[product_id] = product_indices
    
    embeddings = np.array(embeddings).squeeze()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, product_info, product_id_to_indices, clip_model, preprocess, device

def match_products(detections, index, product_info, product_id_to_indices, clip_model, preprocess, device):
    matches = []
    for detection in detections:
        try:
            crop = detection['crop']
            crop_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image_input = preprocess(crop_image).unsqueeze(0).to(device)
            with torch.no_grad():
                query_embedding = clip_model.encode_image(image_input).cpu().numpy()
            distances, indices = index.search(query_embedding, k=1)
            similarity = 1 - distances[0][0] / 2
            match_type = "exact" if similarity > 0.9 else "similar" if similarity > 0.75 else "no_match"
            
            # Find the product_id for the matched index
            matched_index = indices[0][0]
            matched_product_id = None
            for product_id, idx_list in product_id_to_indices.items():
                if matched_index in idx_list:
                    matched_product_id = product_id
                    break
            
            # Get product info
            product = next((p for p in product_info if p['id'] == matched_product_id), None)
            if product:
                matches.append({
                    "type": detection['class'],
                    "color": extract_color(product['product_tags']),
                    "match_type": match_type,
                    "matched_product_id": str(matched_product_id),
                    "confidence": float(similarity)
                })
            else:
                matches.append({
                    "type": detection['class'],
                    "color": "unknown",
                    "match_type": "no_match",
                    "matched_product_id": None,
                    "confidence": 0.0
                })
        except Exception as e:
            print(f"Error matching product for detection: {e}")
            matches.append({
                "type": detection['class'],
                "color": "unknown",
                "match_type": "no_match",
                "matched_product_id": None,
                "confidence": 0.0
            })
    return matches

def extract_color(product_tags):
    if pd.isna(product_tags):
        return "unknown"
    tags = product_tags.lower().split(',')
    for tag in tags:
        if tag.startswith("colour:"):
            return tag.replace("colour:", "").strip().capitalize()
    return "unknown"

def classify_vibe(caption, product_info):
    nlp = spacy.load("en_core_web_sm")
    vibe_keywords = {
        'Coquette': ['lace', 'pink', 'bow', 'floral'],
        'Clean Girl': ['minimal', 'neutral', 'sleek'],
        'Cottagecore': ['vintage', 'pastel', 'nature'],
        'Streetcore': ['urban', 'sneakers', 'graffiti'],
        'Y2K': ['glitter', 'metallic', 'retro'],
        'Boho': ['fringe', 'earthy', 'flowy'],
        'Party Glam': ['sparkle', 'sequin', 'bold']
    }
    text = caption.lower()
    for product in product_info:
        text += f" {product['description'].lower()} {product['product_tags'].lower()}"
    doc = nlp(text)
    scores = {vibe: 0 for vibe in vibe_keywords}
    for token in doc:
        for vibe, keywords in vibe_keywords.items():
            if token.text in keywords:
                scores[vibe] += 1
    return [vibe for vibe, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3] if score > 0] or ["Unknown"]

# Define paths
video_path = "data/sample_video2.mp4"
output_dir = "frames"
detectedframepath = "detected_frames"
images_csv = "data/images.csv"  # Update with actual path
product_data_csv = "data/product_data.csv"  # Update with actual path
caption = '''Ofcourse I’ll get you flowers 🙆🏻‍♀️🙂‍↕️

Spinning into summer with my favorite @virgio.official dress, you like it too? I got you girlie, comment ‘Link’ and I will slide into your dms with the link 🤜🤛

Use code ‘SUKRUTIAIRI’ and save some extra 💸

Location- @roasterycoffeehouseindia 📍Noida

#grwm #summer #summerfit #dress #date #datedress #outfit #fashion #outﬁtinspo
'''  # Update with actual caption

video_id = "sample_video2"
output_json_path = f"outputs/output_{video_id}.json"

# Load YOLO model
try:
    model = YOLO("D:/Aadit/ML/Flickd/runs/detect/train3/weights/best.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Setup FAISS index
try:
    index, product_info, product_id_to_indices, clip_model, preprocess, device = setup_faiss_index(images_csv, product_data_csv)
except Exception as e:
    print(f"Error setting up FAISS index: {e}")
    exit(1)

# Process video
if extract_frames(video_path, output_dir):
    detections = []
    os.makedirs(detectedframepath, exist_ok=True)
    for frame_path in os.listdir(output_dir):
        if frame_path.endswith(".jpg"):
            frame_number = int(frame_path.split("_")[1].split(".")[0])
            full_frame_path = os.path.join(output_dir, frame_path)
            frame_detections = detect_objects(full_frame_path, frame_number, model, detectedframepath)
            detections.extend(frame_detections)
    
    # Match products
    matches = match_products(detections, index, product_info, product_id_to_indices, clip_model, preprocess, device)
    
    # Classify vibe
    vibes = classify_vibe(caption, product_info)
    
    # Format output
    output = {
        "video_id": video_id,
        "vibes": vibes,
        "products": matches
    }
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved output to {output_json_path}")
else:
    print("Failed to extract frames. Exiting.")
    exit(1)