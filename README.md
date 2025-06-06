# Flickd - AI-Powered Fashion Detection and Matching

Flickd is an AI-powered system that detects fashion items in videos and matches them with similar products from a catalog. It uses state-of-the-art computer vision and natural language processing techniques to provide accurate fashion item detection and matching.

## Features

- Video frame extraction and processing
- Fashion item detection using YOLOv8
- Product matching using CLIP embeddings and FAISS indexing
- Vibe classification for fashion items
- Efficient caching system for product embeddings
- Comprehensive logging and error handling

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adi-devv/Flickd-AI-Hackathon.git
cd Flickd-AI-Hackathon
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the YOLOv8 model:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Project Structure

```
Flickd/
├── main.py                 # Main processing script
├── data/
│   ├── cache/             # Cached embeddings and indices
│   └── product_data.csv   # Product catalog data
├── detected_frames/       # Processed video frames with detections
├── outputs/              # JSON output files
├── frames/              # Extracted video frames
└── requirements.txt     # Project dependencies
```

## Usage

1. Prepare your input:
   - Video file for processing
   - Product catalog CSV files
   - Optional vibe taxonomy for classification

2. Run the main script:
```bash
python main.py --video_path path/to/video.mp4 --images_csv path/to/images.csv --product_data_csv path/to/products.csv --caption "Your video caption" --video_id "unique_id" --output_json_path path/to/output.json
```

## Output

The system generates:
- Detected frames with bounding boxes
- JSON output containing:
  - Detected fashion items
  - Matched products
  - Confidence scores
  - Vibe classifications

## Performance

- Uses YOLOv8 for fast and accurate object detection
- Implements FAISS for efficient similarity search
- Caches embeddings to improve processing speed
- Supports batch processing of video frames

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
