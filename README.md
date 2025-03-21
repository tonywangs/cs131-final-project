# MultiSport: Real-time multi-player tracking and bird’s eye analytics for sports footage

Project for the Stanford CS131 (Computer Vision: Foundations and Applications) Final (Winter 2025)

## Inspiration
Sports teams are increasingly turning to data analytics to gain a competitive edge. But tracking player movement remains a bottleneck: existing tools are expensive and often require manual annotation. We wanted to democratize sports analytics with a fast, open-source, deep learning tool that can analyze game footage from a single camera and output detailed player tracking insights automatically.

## What it does
MultiAthlete is a real-time computer vision pipeline that:
- Detects all players in each frame using YOLOv8
- Assigns persistent IDs using SORT (Kalman Filter + IoU-based matching)
- Projects player locations to a bird’s eye view via homography transformation
- Annotates every frame with bounding boxes and player identities
- Reconstructs the video with tracking overlays and mini-map
- Runs entirely on CPU/GPU in-browser via a Streamlit app
Use cases include post-game performance analysis, tactical visualization, and even live feedback during practice.

## How we built it
- Detection: YOLOv8 pretrained on the COCO dataset (person class)
- Tracking: SORT algorithm with Kalman filtering and Hungarian matching
- Homography: 3x3 perspective transform using four field reference points
- Interface: Interactive Streamlit web app for drag-and-drop video uploads
- Pipelining: Modular architecture in Python with OpenCV and NumPy
- Output: Annotated .mp4 video with frame-by-frame identity overlays and bird’s eye projection
We also implemented helper scripts to convert video to frames, calibrate fields, and recompile the final output.

## Challenges we ran into
- Maintaining identity through occlusions, especially when players overlap or wear similar uniforms
- Getting the homography transformation to align perfectly with real-world field dimensions
- Deploying everything as a browser-based app that runs on local hardware without GPU dependency
- Processing video at a speed that’s fast enough for near real-time performance

## Accomplishments that we're proud of
- Building a full tracking pipeline from scratch in just a few weeks
- Achieving real-time performance on a modest GPU
- Creating an intuitive web app that runs end-to-end on user-uploaded video
- Seeing working tracking overlays + bird’s eye projections that coaches could actually use

## What we learned
- The simplicity of YOLOv8 for high-speed detection, and the power of SORT for ID tracking
- How to calculate and apply homographies to map 2D video to field coordinates
- The tradeoffs between speed vs. accuracy in tracking systems

## What's next for MultiAthlete
- Integrate OCR to extract jersey numbers for player name recognition
- Upgrade to OC-SORT or ByteTrack for better occlusion handling
- Add pose estimation (OpenPose or MediaPipe) to extract player actions (e.g. kicking, jumping)
- Build a live dashboard for coaches to view analytics during practice or games
- Deploy the system on edge devices or the cloud for scalable inference

## Built With
- Python
- YOLOv8
- SORT
- OpenCV
- NumPy
- Streamlit
- ffmpeg
- Matplotlib
- Jupyter Notebook
