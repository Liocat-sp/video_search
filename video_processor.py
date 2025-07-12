import streamlit as st
import cv2
import ultralytics
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from db import create_video_frames, delete_videos

yoloModel = ultralytics.YOLO("yolov8n.pt")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def closest_color_name(rgb):
    xkcd_colors = mcolors.XKCD_COLORS

    min_dist = float('inf')
    closest = "Unknown"

    for name, hex_code in xkcd_colors.items():
        r, g, b = [int(x * 255) for x in mcolors.to_rgb(hex_code)]
        dist = sum((a - b) ** 2 for a, b in zip(rgb, (r, g, b)))
        if dist < min_dist:
            min_dist = dist
            closest = name.replace("xkcd:", "")

    return closest


def get_color_percentages(image, n_clusters=5):
    img = image.reshape(-1, 3)

    brightness = np.mean(img, axis=1)
    mask = (brightness > 40) & (brightness < 245)
    filtered = img[mask]
    if len(filtered) == 0:
        return {"Unknown": 100}

    kmeans = KMeans(n_clusters=min(n_clusters, len(filtered)))
    kmeans.fit(filtered)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    total = np.sum(counts)

    color_percentages = {}
    for label, count in zip(labels, counts):
        rgb = kmeans.cluster_centers_[label].astype(int)
        color_name = closest_color_name(rgb)
        percent = round((count / total) * 100, 1)
        color_percentages[color_name] = color_percentages.get(color_name, 0) + percent

    return dict(sorted(color_percentages.items(), key=lambda x: -x[1]))




def process_video_frame(frame_number, video_cap):

    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number);
    ret, frame = video_cap.read()

    if not ret:
        st.error(f"Error reading frame {frame_number} from video.")
        return None;

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = yoloModel(rgb_frame)

    frame_data = [];

    for result in results:
        grouping = {}

        for i, box in enumerate(result.boxes):
            label = result.names[int(box.cls[0].cpu().numpy())]

            conf = box.conf[0].item()

            if conf < 0.7:
                continue;

            grouping[label] = grouping.get(label, 0) + 1

            xyxy = [int(x) for x in box.xyxy[0].cpu().numpy()]  # Ensure list of ints for JSON serializability
            x1, y1, x2, y2 = xyxy

            cropped_frame = rgb_frame[y1:y2, x1:x2]

            colors = get_color_percentages(cropped_frame)

            color_name = ", ".join(colors.keys())

            text = f" {label} with {color_name} colors."

            frame_data.append({
                "label": label,
                "xyxy": xyxy,
                "confidence": conf,
                "colors": {str(k): float(v) for k, v in colors.items()},  # Ensure keys are str, values are float
                "text": str(text)
            })

        annotated_frame = result.plot()
        # st.image(annotated_frame, caption=f"Frame {frame_number} - with detections colors: {colors}")


    if len(frame_data) == 0:
        return None;

    frame_description = ""

    for label, count in grouping.items():
        if frame_description:
            frame_description += ", "
        frame_description += f"{count} {label}(s)"        
    
    frame_description += " .";

    for data in frame_data:
        text = data["text"]

        frame_description += text;


    # print(frame_description);

    data = {
        "frame_number": frame_number,
        "annotated_frame": annotated_frame,
        "description": frame_description,
        "frame_data": frame_data,
        "embedding": embedder.encode(frame_description).tolist()
    }

    create_video_frames(frame_number, frame_data, data['embedding'])

    return data;


def process_video(video_path):
    delete_videos()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    for sec in range(int(frame_count // fps)):
        frame_number = int(sec * fps)
        process_video_frame(frame_number, cap)

    print(f"Total frames in video: {frame_count}")