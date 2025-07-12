import streamlit as st
import tempfile
from db import delete_videos
from video_processor import process_video
from video_search import search_text
import pathlib


def load_style():
    with open(pathlib.Path('style.css')) as f:
        s = f.read();
        st.html(f"<style>{s}</style>")


load_style()

st.write("# Welcome to the Video Search App.");

upload_video_file = st.file_uploader("Upload a video file", accept_multiple_files=False, type=["mp4", "avi", "mov", "mkv"])


if upload_video_file is not None:
    tFile = tempfile.NamedTemporaryFile(delete=False)
    tFile.write(upload_video_file.read())

    video = st.empty() 
    video.video(tFile.name, format="video/mp4", start_time=0)
    
    if "video_path" not in st.session_state or st.session_state["video_path"] != upload_video_file.name:
        if "initialized" in st.session_state:
            del st.session_state["initialized"]
        st.session_state["processed_video"] = process_video(tFile.name)
        st.session_state["video_path"] = upload_video_file.name


    timeline = st.empty()
    cols = st.columns(5)        

    txt = st.text_area("Enter a text query to search in the video", height=100)
    if txt:
        st.write(f"You entered: {txt}")

        def change_video(time):
            st.session_state["initialized"] = time
            video.video(tFile.name, format="video/mp4", start_time=time, autoplay=True)

        result = search_text(txt, tFile.name, cols, timeline, change_video)

        if "initialized" not in st.session_state and result and isinstance(result, list) and "time_in_seconds" in result[0]:
            first_time = result[0]["time_in_seconds"]
            st.session_state["initialized"] = first_time

            video.video(tFile.name, format="video/mp4", start_time=st.session_state["initialized"])

