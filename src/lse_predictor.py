import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import json

class LSEPredictor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Cargar mapeo de clases desde archivo
        self.letter_map = self.load_class_mapping()
        
        self.model = self.load_model()
   
    def load_class_mapping(self):
        """Carga el mapeo de clases desde el archivo JSON generado por train_model.py"""
        mapping_path = os.path.join('models', 'class_mapping.json')
        
        try:
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    loaded_mapping = json.load(f)
                    print(f"✅ Mapping de clases cargado: {loaded_mapping}")
                    return loaded_mapping
            else:
                print("❌ Archivo class_mapping.json no encontrado")
                return {}
        except Exception as e:
            print(f"❌ Error cargando mapping: {e}")
            return {}
   
    def load_model(self):
        """Carga el modelo entrenado - AHORA ES OBLIGATORIO"""
        model_path = os.path.join('models', 'lse_model.h5')
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                print("✅ Modelo LSE cargado exitosamente")
                return model
            else:
                raise FileNotFoundError("❌ Modelo no encontrado. Ejecuta primero: python models/train_model.py")
        except Exception as e:
            raise Exception(f"❌ Error al cargar el modelo: {str(e)}")
   
    def extract_landmarks(self, frame):
        """Extrae landmarks de las manos usando MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                return np.array(landmarks), hand_landmarks
            return None, None
        except Exception as e:
            print(f"Error extrayendo landmarks: {e}")
            return None, None
   
    def create_landmarks_object(self, landmarks_data):
        """Crea un objeto landmarks a partir de los datos"""
        return landmarks_data
   
    def predict_letter(self, frame):
        """Predice la letra basada en los landmarks - SOLO MODO REAL"""
        try:
            landmarks, landmarks_data = self.extract_landmarks(frame)
            
            if landmarks is not None:
                # Preprocesar landmarks
                landmarks = landmarks.reshape(1, -1)
                
                # Predecir con el modelo real
                prediction = self.model.predict(landmarks, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Convertir a string para el mapping
                predicted_class_str = str(predicted_class)
                letter = self.letter_map.get(predicted_class_str, '?')
                
                return letter, confidence, landmarks_data
            
            # No se detectaron manos
            return "No hands", 0.0, None
            
        except Exception as e:
            print(f"❌ Error en predict_letter: {e}")
            return "Error", 0.0, None