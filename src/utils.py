import cv2
import numpy as np

def draw_hand_landmarks(frame, landmarks):
    """Dibuja los landmarks de la mano en el frame"""
    if landmarks is not None:
        h, w, _ = frame.shape
        landmarks = landmarks.reshape(-1, 3)
        
        for i, (x, y, z) in enumerate(landmarks):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
   
    return frame

def create_bbox_from_landmarks(landmarks, frame_shape, padding=20):
    """Crea un bounding box alrededor de los landmarks"""
    if landmarks is None:
        return None
    
    landmarks = landmarks.reshape(-1, 3)
    h, w = frame_shape[:2]
    
    x_coords = landmarks[:, 0] * w
    y_coords = landmarks[:, 1] * h
    
    x_min = max(0, int(np.min(x_coords) - padding))
    y_min = max(0, int(np.min(y_coords) - padding))
    x_max = min(w, int(np.max(x_coords) + padding))
    y_max = min(h, int(np.max(y_coords) + padding))
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)