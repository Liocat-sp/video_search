import base64
from io import BytesIO
import json
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit as st
from db import search_video_frame
import cv2
from streamlit_clickable_images import clickable_images

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def search_text(text, video_path, cols, timeline, change_video):
    embedding = embedder.encode(text).tolist()

    results = search_video_frame(embedding)   

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video file.")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    final_result = []

    timeline.markdown("### Timeline View")

    for i, result in enumerate(results):
        cap.set(cv2.CAP_PROP_POS_FRAMES, result[1])
        
        ret, frame = cap.read()        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        time_in_seconds = result[1] / fps

        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        if isinstance(result[3], str):
            data = json.loads(result[3])
        else:
            data = result[3]

        # confidence = ",".join([f"{d['confidence']:.2f}" for d in data])

        with cols[i]:
                
            st.image(rgb_frame, use_container_width=True)

            if st.button(f"`{time_str}`", key=f"time-btn-{i}"):
                change_video(time_in_seconds)


        final_result.append({
            'time_in_seconds': time_in_seconds,
            'frame_number': result[1]
        })


    return final_result




