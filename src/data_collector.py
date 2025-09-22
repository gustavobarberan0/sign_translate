import cv2
import os
import numpy as np
from datetime import datetime
import json
import mediapipe as mp

class LSEDataCollector:
    def __init__(self, data_dir='data/collected_data'):
        self.data_dir = data_dir
        self.current_letter = 'A'
        self.counter = 0
        
        # Inicializar MediaPipe para extraer landmarks
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.setup_directories()
        
    def setup_directories(self):
        """Crea directorios para cada letra del abecedario"""
        # Crear directorio principal si no existe
        os.makedirs(self.data_dir, exist_ok=True)
        
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for letter in letters:
            os.makedirs(f'{self.data_dir}/{letter}', exist_ok=True)
        print("Directorios creados para todas las letras")
    
    def extract_landmarks(self, frame):
        """Extrae landmarks de un frame usando MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            return landmarks
        return None
    
    def collect_samples(self, letter, num_samples=100):
        """Recolecta muestras para una letra específica y guarda landmarks en JSON"""
        self.current_letter = letter.upper()
        self.counter = 0
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se puede acceder a la cámara")
            return
        
        print(f"Recolectando {num_samples} muestras para la letra {self.current_letter}")
        print("Presiona 'c' para capturar, 'q' para salir")
        
        while self.counter < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mostrar instrucciones
            cv2.putText(frame, f"Letra: {self.current_letter}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Muestras: {self.counter}/{num_samples}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Presiona 'c' para capturar", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dibujar área de interés
            cv2.rectangle(frame, (220, 100), (420, 380), (0, 255, 0), 2)
            
            cv2.imshow('Recolector de Datos LSE', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Extraer landmarks
                landmarks = self.extract_landmarks(frame)
                
                if landmarks and len(landmarks) == 63:  # 21 landmarks * 3 coordenadas
                    # Guardar landmarks en JSON
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    json_filename = f"{self.data_dir}/{self.current_letter}/{timestamp}.json"
                    
                    with open(json_filename, 'w') as f:
                        json.dump(landmarks, f)
                    
                    # También guardar imagen para referencia (opcional)
                    img_filename = f"{self.data_dir}/{self.current_letter}/{timestamp}.jpg"
                    cv2.imwrite(img_filename, frame)
                    
                    self.counter += 1
                    print(f"Capturada muestra {self.counter}/{num_samples}")
                else:
                    print("Error: No se detectaron landmarks. Intenta nuevamente.")
                    
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Recolección para {self.current_letter} completada")

if __name__ == "__main__":
    collector = LSEDataCollector()
    collector.collect_samples('A', 10)  # Probar con 10 muestras