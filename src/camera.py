import cv2

class CameraManager:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            # Probar con índice 1 si el 0 falla
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise Exception("No se puede acceder a la cámara")
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
   
    def release(self):
        if self.cap:
            self.cap.release()