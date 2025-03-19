import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO  # For player detection

def create_birds_eye_view(frame, court_corners=None):
    """
    Convert a basketball court frame to a birds-eye view with player positions mapped.
    
    Args:
        frame: The input video frame (numpy array)
        court_corners: Optional list of 4 points defining the court corners in the frame.
                      If None, will prompt user to select corners.
    
    Returns:
        A birds-eye view visualization of the court with player positions
    """
    # Make a copy of the frame to work with
    img = frame.copy()
    h, w = img.shape[:2]
    
    # Step 1: Define the court corners if not provided
    if court_corners is None:
        # For simplicity, we'll use a default court rectangle
        # In practice, you would use a more sophisticated approach like court line detection
        # or allow the user to mark the corners manually
        court_corners = np.array([
            [int(w * 0.2), int(h * 0.15)],  # Top-left
            [int(w * 0.8), int(h * 0.15)],  # Top-right
            [int(w * 0.8), int(h * 0.85)],  # Bottom-right
            [int(w * 0.2), int(h * 0.85)]   # Bottom-left
        ], dtype=np.float32)
    
    # Step 2: Define the destination points for the birds-eye view
    # Standard basketball court dimensions: 94 feet x 50 feet
    court_width = 500    # Width in pixels for our birds-eye view
    court_height = 940   # Height in pixels for our birds-eye view
    
    dst_points = np.array([
        [0, 0],
        [court_width, 0],
        [court_width, court_height],
        [0, court_height]
    ], dtype=np.float32)
    
    # Step 3: Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(court_corners, dst_points)
    
    # Step 4: Detect players using YOLO
    model = YOLO('yolov8n.pt')  # Load a pre-trained YOLO model
    results = model(img)
    
    # Filter for person class (class 0 in COCO dataset)
    boxes = []
    for result in results:
        detections = result.boxes
        for i, box in enumerate(detections):
            cls = int(box.cls.item())
            if cls == 0:  # Person class
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    # Step 5: Get player positions (bottom center of bounding boxes)
    player_positions = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Bottom center point of the bounding box
        foot_position = [int((x1 + x2) / 2), y2]
        player_positions.append(foot_position)
    
    # Step 6: Create a birds-eye view
    court_img = np.ones((court_height, court_width, 3), dtype=np.uint8) * 255
    
    # Draw court lines (simplified)
    # Outer boundary
    cv2.rectangle(court_img, (0, 0), (court_width, court_height), (0, 0, 0), 2)
    
    # Half-court line
    cv2.line(court_img, (0, court_height // 2), (court_width, court_height // 2), (0, 0, 0), 2)
    
    # Center circle
    center_x, center_y = court_width // 2, court_height // 2
    cv2.circle(court_img, (center_x, center_y), 60, (0, 0, 0), 2)
    
    # Free throw circles and lines
    # Top
    cv2.circle(court_img, (center_x, int(court_height * 0.15)), 60, (0, 0, 0), 2)
    cv2.line(court_img, (int(court_width * 0.25), int(court_height * 0.15)), 
             (int(court_width * 0.75), int(court_height * 0.15)), (0, 0, 0), 2)
    
    # Bottom
    cv2.circle(court_img, (center_x, int(court_height * 0.85)), 60, (0, 0, 0), 2)
    cv2.line(court_img, (int(court_width * 0.25), int(court_height * 0.85)), 
             (int(court_width * 0.75), int(court_height * 0.85)), (0, 0, 0), 2)
    
    # Step 7: Transform player positions to birds-eye view coordinates
    birds_eye_positions = []
    for pos in player_positions:
        # Add a 1 for homogeneous coordinates
        pos_homogeneous = np.array([pos[0], pos[1], 1])
        
        # Apply perspective transformation
        transformed = M.dot(pos_homogeneous)
        
        # Normalize
        transformed = transformed / transformed[2]
        birds_eye_positions.append((int(transformed[0]), int(transformed[1])))
    
    # Step 8: Draw players on the birds-eye view
    for i, pos in enumerate(birds_eye_positions):
        # Skip players outside the court
        if (0 <= pos[0] < court_width and 0 <= pos[1] < court_height):
            # Draw as a circle
            cv2.circle(court_img, pos, 10, (255, 0, 0), -1)  # Blue circles for players
            
            # Add player number
            cv2.putText(court_img, str(i+1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw the original frame with detected players for reference
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        foot_position = [int((x1 + x2) / 2), y2]
        cv2.circle(img, tuple(foot_position), 5, (0, 0, 255), -1)
    
    return img, court_img

def main():
    # Example usage:
    # 1. Load a video frame
    frame = cv2.imread('basketball_frame.jpg')
    
    # 2. Process the frame
    original_with_detection, birds_eye = create_birds_eye_view(frame)
    
    # 3. Display results
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Frame with Player Detection')
    plt.imshow(cv2.cvtColor(original_with_detection, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Birds-Eye View')
    plt.imshow(court_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save output images
    cv2.imwrite('player_detection.jpg', original_with_detection)
    cv2.imwrite('birds_eye_view.jpg', birds_eye)

if __name__ == "__main__":
    main()