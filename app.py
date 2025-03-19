import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time

# If you are using YOLOv8:
from ultralytics import YOLO

# If you have SORT or another tracker:
# pip install sort-tracker
# Or if you have a local sort.py, import it.
from sort_tracker import Sort  # Adjust import as needed

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
        return 0, 0
    
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
    Converts a folder of frames to a video file.
    """
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.startswith("frame_")])
    if not frame_files:
        raise ValueError("No frames found in the folder to convert to video.")

    # Load first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width = first_frame.shape[:2]

    # Try a different codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from 'mp4v' to 'avc1'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames directly without storing them all in memory
    for file in frame_files:
        frame_path = os.path.join(frames_folder, file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)

    out.release()

    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        st.success(f"Video saved to {output_video_path}")
        return output_video_path
    else:
        raise ValueError("Video file was not created successfully")


def process_frames_with_tracking(
    input_folder, 
    model_path="yolov8n.pt",
    output_folder=None,
    conf_thres=0.5,
    max_frames=None,
    fps=30
):
    """
    Process frames with YOLOv8 to detect people, track them across frames,
    and create a folder of processed frames with assigned IDs.

    Parameters:
    - input_folder: Folder containing the input frames
    - model_path: path to the YOLOv8 model
    - output_folder: Folder in which to save processed frames (if None, overwrite)
    - conf_thres: confidence threshold for YOLO
    - max_frames: Maximum number of frames to process (for testing)
    - fps: Just for reference if needed
    """

    # Initialize YOLO model
    model = YOLO(model_path)

    # Initialize SORT tracker
    tracker = Sort()

    # Make sure we have a place to store processed frames
    if output_folder is None:
        output_folder = input_folder

    # Get frame files
    frame_files = sorted([f for f in os.listdir(input_folder) if f.startswith("frame_")])
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
    
    if not frame_files:
        st.error(f"No frames found in {input_folder}")
        return
    
    # Process frames
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_folder, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            st.warning(f"Could not read {frame_path}")
            continue

        # Run YOLOv8 inference
        results = model(image, conf=conf_thres)

        # Collect detection boxes for SORT: [x1, y1, x2, y2, confidence]
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Check if detection is person (class 0 in COCO)
                if cls == 0:  # person class
                    detections.append([x1, y1, x2, y2, conf])

        # Convert detections to NumPy array
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # Update tracker
        tracked_objects = tracker.update(detections)

        # Draw tracking results on the frame
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw ID
            cv2.putText(
                image, 
                f"ID: {track_id}", 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

            # Optional: If you wanted to do OCR on the player's jersey, you could add that here:
            # ...
        
        # Save the processed frame (overwriting or saving to output_folder)
        processed_frame_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(processed_frame_path, image)


####################
# Streamlit Layout #
####################

st.title("CS131 Final Project: Sports Analytics Demo (Computer Vision + SORT Tracking)")

st.markdown(
    """
This is our demo for our CS131 Winter 2025 Final Project. This demo lets you:
1. Upload a sports video.
2. Convert it to frames.
3. Run a YOLO + SORT tracking pipeline to detect players and assign IDs.
4. Convert processed frames back into a video.
"""
)

# Step 1: Upload video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    # Initialize session state if not already done
    if 'detection_complete' not in st.session_state:
        st.session_state.detection_complete = False

    # We store the file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    temp_video_path = tfile.name
    
    # Create a unique temp directory to store frames
    frames_dir = tempfile.mkdtemp()
    st.write(f"Debug - Frames directory: {frames_dir}")

    st.write("**Converting video to frames...**")
    frame_count, fps = video_to_frames(temp_video_path, frames_dir)
    if frame_count == 0:
        st.stop()
    st.write(f"Extracted {frame_count} frames at {fps} FPS.")

    # Detection + Tracking options
    run_detection = st.button("Detect & Track Players with YOLO + SORT")

    if run_detection:
        with st.spinner("Detecting and tracking players in frames..."):
            # Create a separate directory for processed frames
            processed_frames_dir = os.path.join(tempfile.gettempdir(), "processed_frames")
            if not os.path.exists(processed_frames_dir):
                os.makedirs(processed_frames_dir)
            
            process_frames_with_tracking(
                input_folder=frames_dir, 
                model_path="yolov8n.pt", 
                output_folder=processed_frames_dir,  # Save to new directory
                conf_thres=0.5,
                max_frames=None,
                fps=fps
            )
        st.session_state.detection_complete = True
        st.session_state.processed_frames_dir = processed_frames_dir  # Store the path
        st.success("Detection and tracking complete!")
        
        # Debug: Show a processed frame
        frame_files = sorted([f for f in os.listdir(processed_frames_dir) if f.startswith("frame_")])
        if frame_files:
            processed_frame = cv2.imread(os.path.join(processed_frames_dir, frame_files[0]))
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.write("Debug - Sample processed frame after detection:")
            st.image(processed_frame_rgb, caption="Sample processed frame with bounding boxes")

    # Step 3: Convert frames back to video
    reconstruct_button = st.button("Reconstruct Processed Video")
    if reconstruct_button:
        if not st.session_state.detection_complete:
            st.warning("Please run detection and tracking first!")
        else:
            output_video_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
            
            # Use the processed frames directory instead of original frames
            frames_dir_for_video = st.session_state.processed_frames_dir
            
            # Debug: Show what frames we're using for video creation
            st.write("Debug - Checking frames before video creation:")
            frame_files = sorted([f for f in os.listdir(frames_dir_for_video) if f.startswith("frame_")])
            st.write(f"Number of frames found: {len(frame_files)}")
            st.write("First few frame names:", frame_files[:3])
            
            if frame_files:
                sample_frame = cv2.imread(os.path.join(frames_dir_for_video, frame_files[0]))
                sample_frame_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                st.write("Debug - Sample frame being used for video creation:")
                st.image(sample_frame_rgb, caption="Frame that will be used in video")
                st.write(f"Frame dimensions: {sample_frame.shape}")

            with st.spinner("Reconstructing video from frames..."):
                frames_to_video(frames_dir_for_video, output_video_path, fps=fps)
            st.success("Video reconstruction complete!")

            # Display the resulting video
            st.video(output_video_path)

            # # Provide a download link
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )
