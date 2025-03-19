import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time

from ultralytics import YOLO
# If you have SORT or other tracking logic, import your tracker here:
# from sort import *

#####################
# Utility functions #
#####################

def video_to_frames(video_path, output_folder):
    """
    Converts a video file to individual frames.

    Args:
        video_path (str): Path to input video.
        output_folder (str): Folder to save frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    st.write(f"**Video Properties**: {width}x{height}, {fps} FPS, {frame_count} frames.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1
    
    cap.release()
    return count, fps


def frames_to_video(frames_folder, output_video_path, fps=30):
    """
    Converts a folder of frames back into a video file.

    Args:
        frames_folder (str): Folder containing frames.
        output_video_path (str): Path for the output video.
        fps (float): Desired output FPS.
    """
    # Gather frame files
    frame_files = sorted(
        [f for f in os.listdir(frames_folder) if f.startswith("frame_")]
    )
    if not frame_files:
        raise ValueError("No frames found in the folder to convert to video.")

    # Read the first frame to get shape
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for file in frame_files:
        frame_path = os.path.join(frames_folder, file)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    return output_video_path


def detect_players_in_frames(frames_folder, model_path="yolov8n.pt", conf_thres=0.5):
    """
    Run person detection on all frames in a folder using a YOLO model.

    Args:
        frames_folder (str): Path to folder containing frames.
        model_path (str): Path to the YOLO model (e.g. "yolov8n.pt").
        conf_thres (float): Confidence threshold for detections.
    """
    model = YOLO(model_path)
    frame_files = sorted(
        [f for f in os.listdir(frames_folder) if f.startswith("frame_")]
    )

    # Optional: If you have a SORT or other tracker, initialize here
    # tracker = Sort()

    for file in frame_files:
        frame_path = os.path.join(frames_folder, file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Inference
        results = model(frame, conf=conf_thres)

        # Draw bounding boxes on the frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # YOLOv8 'person' class is usually 0
                if cls_id == 0:
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Save the processed frame (overwriting or in new folder)
        cv2.imwrite(frame_path, frame)


####################
# Streamlit Layout #
####################

st.title("Sports Analytics Demo (Computer Vision)")

st.markdown(
    """
This demo lets you:
1. Upload a sports video.
2. Convert it to frames.
3. Run a YOLO person detection pipeline.
4. Convert processed frames back into a video.
"""
)

# Step 1: Upload video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    # We store the file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    temp_video_path = tfile.name
    
    # Create a unique temp directory to store frames
    frames_dir = tempfile.mkdtemp()

    st.write("**Converting video to frames...**")
    frame_count, fps = video_to_frames(temp_video_path, frames_dir)
    st.write(f"Extracted {frame_count} frames at {fps} FPS.")

    # Detection options
    st.write("**Run Person Detection**")
    run_detection = st.button("Detect Players")

    if run_detection:
        with st.spinner("Detecting players in frames..."):
            # You can specify your own YOLO model here
            detect_players_in_frames(frames_dir, model_path="yolov8n.pt", conf_thres=0.5)
        st.success("Detection complete!")

    # Step 3: Convert frames back to video
    if st.button("Reconstruct Processed Video"):
        output_video_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
        with st.spinner("Reconstructing video from frames..."):
            frames_to_video(frames_dir, output_video_path, fps=fps)
        st.success("Video reconstruction complete!")

        # Display the resulting video
        st.video(output_video_path)

        # Provide a download link
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_output.mp4",
                mime="video/mp4"
            )