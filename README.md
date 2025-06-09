# Flickd AI Hackathon: Smart Tagging & Vibe Classification Engine

**Submission for the Flickd AI Hackathon**
**By: [Your Name Here]**

---

## 🚀 Project Overview

This project is a fully working Minimum Viable Product (MVP) of Flickd’s "Smart Tagging & Vibe Classification Engine." The engine analyzes short fashion videos and images to automatically identify products and classify the video's "vibe," forming the AI backbone for Flickd's scroll-native, video-first shopping experience.

The system takes a media file (video or image) as input and outputs structured JSON data containing detected products matched against a catalog and a list of relevant fashion vibes.

## ✨ Core Features

-   **Video & Image Processing**: Ingests both `.mp4` videos and `.jpg`/`.png` images.
-   **Object Detection**: Utilizes a pre-trained **YOLOv8** model to detect general object classes like 'person' and 'handbag' that contain fashion items.
-   **Product Matching**: Employs a powerful **CLIP** model to generate image embeddings for detected items and matches them against a pre-indexed product catalog using **FAISS** for high-speed similarity search.
-   **Vibe Classification**: A hybrid NLP system that uses both fast rule-based keyword matching and a sophisticated **Hugging Face Transformer** (zero-shot classification) to assign 1-3 relevant vibes from a predefined list.
-   **Interactive Web Demo**: A user-friendly web UI built with **Gradio** that allows for easy drag-and-drop testing and visualizes the results, including a gallery of matched product images.

## 🛠️ Tech Stack & Models

-   **Backend**: Python 3.9+
-   **AI / Machine Learning**:
    -   `PyTorch`: Core deep learning framework.
    -   `ultralytics`: For running the YOLOv8 object detection model.
    -   `transformers`: For the CLIP (Image Embedding) and BART (NLP) models.
    -   `faiss-cpu`: For efficient similarity search on embeddings.
-   **Data Handling**: `pandas`, `numpy`, `Pillow`
-   **Video Processing**: `opencv-python`
-   **Web UI**: `gradio`
-   **Pre-trained Models**:
    -   **Object Detection**: `yolov8n.pt`
    -   **Image-to-Product Matching**: `openai/clip-vit-base-patch32`
    -   **Vibe Classification**: `facebook/bart-large-mnli`

## 📁 Project Structure

```
flickd_hackathon/
├── development/
│   ├── data/
│   │   ├── media_inputs/       # Sample videos and images for processing.
│   │   ├── catalog_images/     # All product catalog images.
│   │   ├── text_captions/      # .txt files with captions for videos.
│   │   ├── product_data.xlsx   # The product catalog metadata.
│   │   └── vibes_list.json       # The official list of possible vibes.
│   ├── models/                 # Stores pre-computed index and map files.
│   │   ├── catalog.index
│   │   └── product_map.json
│   ├── outputs/                # Where the final JSON outputs are saved.
│   └── src/
│       ├── build_catalog_index.py # One-time script to build the FAISS index.
|       ├── process_media.py       # Json Output Generation for all the input videos.
│       ├── processing_logic.py    # The core ML pipeline logic.
│       └── app.py                 # The Gradio web application.
├── README.md                   # This file.
└── requirements.txt            # All Python dependencies.
```

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [your-repository-url]
cd flickd_hackathon
```

### 2. Create a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## ⚡ How to Run

The project has two main parts: a one-time setup script and the main web application.

### Step 1: Build the Product Catalog Index (Run Once)

Before running the main application, you must pre-process the product catalog. This script generates the embeddings for all catalog images and builds the FAISS index for fast searching.

```bash
python development/src/build_catalog_index.py
```
This will create `catalog.index` and `product_map.json` inside the `development/models/` directory. You only need to run this script once.

### Step 2: Launch the Web Demo

To start the interactive web UI, run the `app.py` script.

```bash
python development/src/app.py
```

After running, the terminal will display a local URL. Open it in your web browser.

```
Running on local URL:  http://127.0.0.1:7860
```

## 🖥️ Using the Web Demo

The Gradio application provides a simple interface:

1.  **Upload**: Drag and drop a video or image file into the "Upload Video or Image" box on the left.
2.  **Submit**: The analysis will start automatically.
3.  **View Results**:
    -   The right-hand side will display the raw **JSON output** containing the detected products and vibes.
    -   Below the JSON, a **gallery will display the images** of the matched products from the catalog.

## 🏛️ Architecture Overview

The system is designed as a modular pipeline:

1.  **Preprocessing**: The `get_frames_from_media` function handles both video and image inputs, standardizing them into a list of processable frames.
2.  **Parallel Processing**:
    -   **Vision Pipeline**: Frames are passed to YOLOv8 for object detection. Each detected item is cropped and passed to CLIP to generate an embedding, which is then matched against the FAISS index.
    -   **NLP Pipeline**: For videos, the corresponding caption file is read and passed to a hybrid NLP function that uses keyword matching and a zero-shot transformer model to classify vibes.
3.  **Aggregation**: The results from the vision and NLP pipelines are aggregated and formatted into the final, structured JSON output.

## 🔮 Future Improvements

-   **Fine-Tuned Detection Model**: Train a custom YOLOv8 model on a dedicated fashion dataset to directly detect specific classes like `top`, `bottoms`, `jacket`, and `earrings`, improving detection accuracy significantly.
-   **Advanced Attribute Extraction**: Go beyond dominant color by using models to extract other attributes like patterns (`striped`, `floral`), materials (`denim`, `leather`), and styles (`v-neck`, `high-waisted`).
-   **Scalable Deployment**: For a production environment, refactor the application to use a dedicated API service (like **FastAPI**) and a background task queue (like **Celery** with **Redis**) to handle long-processing video files without blocking the UI.
-   **Enhanced Vibe Classification**: Incorporate audio-to-text transcription (e.g., using **Whisper**) to analyze spoken words in addition to written captions, providing more context for vibe classification.