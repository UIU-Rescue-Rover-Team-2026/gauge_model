import cv2
import math
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = '/home/mathinonthemoon/gauge_project/runs/pose/train/weights/best.pt'
CAMERA_INDEX = 0  # Try 0, 1, or 2 if the camera fails to open
MAX_PSI = 80     # Max PSI for the trained dataset, adjust if max different
MAX_ANGLE = 272  # This is the angle difference that corresponds to 80 PSI on the trained dataset

# Smoothing / Moving Average settings
history = []
history_size = 10 #Smooth over the last 10 readings 

def calculate_psi(points):
    """
    Calculates PSI from keypoints: [Pivot, Tip, Zero]
    kpts format: [[x1, y1], [x2, y2], [x3, y3]]
    """
    p, t, z = points[0], points[1], points[2]
    
    # Calculate angles relative to the Pivot point using atan2(y, x)
    angle_zero = math.atan2(z[1] - p[1], z[0] - p[0])
    angle_tip = math.atan2(t[1] - p[1], t[0] - p[0])
    
    # Calculate difference and normalize to 0-360 degrees
    diff_deg = math.degrees(angle_tip - angle_zero) % 360
    
    # Map degrees to PSI scale
    CALIBRATION_OFFSET = 0 # Adjust this until 80 shows as 80
    raw_psi = (diff_deg / MAX_ANGLE) * MAX_PSI + CALIBRATION_OFFSET
    return max(0, min(raw_psi, MAX_PSI))

# --- INITIALIZATION ---
print("Loading YOLO Model...")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"CRITICAL ERROR: Could not open camera at index {CAMERA_INDEX}")
    exit()

print("System Active. Press 'q' in the window or Ctrl+C in terminal to stop.")

# --- MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Inference: imgsz=480 helps performance on your MX450
    results = model(frame, stream=True, verbose=False, imgsz=480, conf=0.4)
    
    for r in results:
        # STEP 1: Check if keypoints object exists and has data
        # We check .data.shape[0] to ensure at least one object was detected
        if r.keypoints is not None and r.keypoints.data.shape[0] > 0:
            
            try:
                # STEP 2: Safely extract the first detection's points
                kpts = r.keypoints.xy[0].cpu().numpy()

                # STEP 3: Verify we have at least 3 points (Pivot, Tip, Zero)
                if kpts.shape[0] >= 3:
                    # Check that the points aren't just [0, 0] (happens on low-conf detections)
                    if np.any(kpts[0]) and np.any(kpts[1]):
                        
                        # Calculate and smooth the reading
                        current_psi = calculate_psi(kpts)
                        history.append(current_psi)
                        if len(history) > history_size:
                            history.pop(0)
                        smoothed_psi = sum(history) / len(history)

                        # Drawing coordinates
                        pivot = tuple(kpts[0].astype(int))
                        tip = tuple(kpts[1].astype(int))
                        zero = tuple(kpts[2].astype(int))

                        # Draw indicators
                        cv2.line(frame, pivot, zero, (255, 0, 0), 2)  # Blue = Zero Ref
                        cv2.line(frame, pivot, tip, (0, 0, 255), 3)   # Red = Needle
                        cv2.circle(frame, pivot, 5, (0, 255, 255), -1) # Yellow = Pivot
                        
                        # Visual Overlay
                        cv2.putText(frame, f"PSI: {smoothed_psi:.2f}", (30, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        
                        # Terminal Update
                        print(f"\r[DETECTED] Pressure: {smoothed_psi:.2f} PSI   ", end="")
            
            except Exception as e:
                # Catch-all for any unexpected indexing issues within a frame
                continue
        else:
            # Display searching status if no gauge is found
            cv2.putText(frame, "Searching for Gauge...", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("\r[SEARCHING] No gauge in view...             ", end="")

    # --- OUTPUT HANDLING ---
    try:
        cv2.imshow("UIU Rescue Rover - Vision Telemetry", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        # Fallback for Headless Linux environments
        cv2.imwrite('telemetry_debug.jpg', frame)
        pass

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nSystem Shutdown Cleanly.")
