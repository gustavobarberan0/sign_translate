import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import cv2
import os
import sys

# Añadir el directorio src al path para importaciones relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import CameraManager
from src.lse_predictor import LSEPredictor

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignTranslate - Traductor de Lenguaje de Señas Español")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables de control
        self.is_camera_active = False
        self.camera_manager = None
        self.lse_predictor = LSEPredictor()
        self.current_prediction = ""
        self.confidence = 0
        self.show_landmarks = tk.BooleanVar(value=True)
        self.confidence_threshold = tk.DoubleVar(value=0.7)
        
        # Historial de predicciones para suavizado
        self.prediction_history = []
        
        # Configurar la interfaz
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header con logo y título
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Título de la aplicación
        title_label = ttk.Label(header_frame, text="SignTranslate", font=("Arial", 24, "bold"), foreground="#2c3e50")
        title_label.grid(row=0, column=0, padx=(0, 10))
        
        subtitle_label = ttk.Label(header_frame, text="Traductor de Lenguaje de Señas Español",
                                    font=("Arial", 12), foreground="#7f8c8d")
        subtitle_label.grid(row=0, column=1)
        
        # Panel de visualización de video
        video_frame = ttk.LabelFrame(main_frame, text="Vista previa", padding="10")
        video_frame.grid(row=1, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="Cámara desactivada", background="black",
                                    foreground="white", anchor="center")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), ipady=80)
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Letra detectada
        self.prediction_label = ttk.Label(results_frame, text="", font=("Arial", 72, "bold"),
                                         foreground="#2c3e50", anchor="center")
        self.prediction_label.grid(row=0, column=0, pady=(20, 10))
        
        # Barra de confianza
        confidence_frame = ttk.Frame(results_frame)
        confidence_frame.grid(row=1, column=0, pady=(0, 20), sticky=(tk.W, tk.E))
        
        ttk.Label(confidence_frame, text="Confianza:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.StringVar(value="0%")
        ttk.Label(confidence_frame, textvariable=self.confidence_var, font=("Arial", 10, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        self.confidence_bar = ttk.Progressbar(confidence_frame, mode='determinate', length=200)
        self.confidence_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Panel de control
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.toggle_btn = ttk.Button(control_frame, text="Iniciar Cámara", command=self.toggle_camera)
        self.toggle_btn.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(control_frame, text="Recolectar Datos", command=self.open_data_collector).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(control_frame, text="Configuración", command=self.show_settings).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(control_frame, text="Ayuda", command=self.show_help).grid(row=0, column=3)
        
        # Footer
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Label(footer_frame, text="SignTranslate v1.0 (Gustavo Barberan) • Lenguaje de Señas Español • 2025",
                                 font=("Arial", 9), foreground="#95a5a6").grid(row=0, column=0)
        
        # Ajustar tamaños de filas y columnas
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        
    def toggle_camera(self):
        if self.is_camera_active:
            self.stop_camera()
            self.toggle_btn.config(text="Iniciar Cámara")
        else:
            self.start_camera()
            self.toggle_btn.config(text="Detener Cámara")
   
    def start_camera(self):
        try:
            self.camera_manager = CameraManager()
            self.is_camera_active = True
            self.update_frame()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar la cámara: {str(e)}")
            self.is_camera_active = False
   
    def stop_camera(self):
        self.is_camera_active = False
        if self.camera_manager:
            self.camera_manager.release()
            self.camera_manager = None
   
    def update_frame(self):
        if self.is_camera_active and self.camera_manager:
            frame = self.camera_manager.get_frame()
            
            if frame is not None:
                frame = cv2.flip(frame, 1)
                # Dibuja el recuadro de interés para la mano
                cv2.rectangle(frame, (220, 100), (420, 380), (0, 255, 0), 2)
                cv2.putText(frame, "Coloca la mano aqui", (220, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Pasar el fotograma al predictor
                prediction, confidence, landmarks_data = self.lse_predictor.predict_letter(frame)
                
                # Actualizar la interfaz con la predicción (si es válida)
                if prediction is not None:
                    self.update_prediction(prediction, confidence)
                # Dibujar los landmarks si la opción está activada
                if self.show_landmarks.get() and landmarks_data:
                    # Convierte landmarks_data a un objeto que MediaPipe pueda dibujar
                    landmarks = self.lse_predictor.create_landmarks_object(landmarks_data)
                    
                    # Usa el método de MediaPipe para dibujar las conexiones y los puntos
                    self.lse_predictor.mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        self.lse_predictor.mp_hands.HAND_CONNECTIONS,
                        self.lse_predictor.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.lse_predictor.mp_drawing_styles.get_default_hand_connections_style()
                    )
                # Convertir a formato que Tkinter puede mostrar
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
                
            self.root.after(15, self.update_frame)
        else:
            self.video_label.configure(image='')
            self.video_label.config(text="Cámara desactivada")
   
    def update_prediction(self, prediction, confidence):
        # Añadir al historial para suavizado
        self.prediction_history.append((prediction, confidence))
        if len(self.prediction_history) > 5:
            self.prediction_history.pop(0)
        
        # Encontrar la predicción más común
        if self.prediction_history:
            from collections import Counter
            predictions = [p[0] for p in self.prediction_history]
            most_common = Counter(predictions).most_common(1)[0][0]
            
            # Calcular confianza promedio
            confidences = [p[1] for p in self.prediction_history if p[0] == most_common]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            self.current_prediction = most_common
            self.confidence = avg_confidence
        else:
            self.current_prediction = prediction
            self.confidence = confidence
        
        # Actualizar la interfaz
        if self.confidence >= self.confidence_threshold.get():
            self.prediction_label.config(text=self.current_prediction)
            self.confidence_bar['value'] = self.confidence * 100
            self.confidence_var.set(f"{self.confidence * 100:.1f}%")
        else:
            self.prediction_label.config(text="")
            self.confidence_bar['value'] = 0
            self.confidence_var.set("0%")
   
    def open_data_collector(self):
        """Abre la ventana de recolección de datos"""
        from src.data_collector import LSEDataCollector
        collector = LSEDataCollector()
        
        # Ventana para seleccionar letra
        letter_window = tk.Toplevel(self.root)
        letter_window.title("Recolectar Datos")
        letter_window.geometry("300x200")
        
        ttk.Label(letter_window, text="Selecciona la letra a recolectar:",
                                 font=("Arial", 12)).pack(pady=10)
        
        letter_var = tk.StringVar(value="A")
        letter_combo = ttk.Combobox(letter_window, textvariable=letter_var,
                                    values=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        letter_combo.pack(pady=5)
        
        ttk.Label(letter_window, text="Número de muestras:").pack(pady=5)
        samples_var = tk.IntVar(value=50)
        samples_spin = ttk.Spinbox(letter_window, from_=10, to=200,
                                    textvariable=samples_var, width=10)
        samples_spin.pack(pady=5)
        
        def start_collection():
            letter_window.destroy()
            collector.collect_samples(letter_var.get(), samples_var.get())
        
        ttk.Button(letter_window, text="Comenzar", command=start_collection).pack(pady=20)
   
    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Configuración")
        settings_window.geometry("400x200")
        
        ttk.Label(settings_window, text="Opciones de configuración",
                                 font=("Arial", 14, "bold")).pack(pady=10)
        
        ttk.Checkbutton(settings_window, text="Mostrar puntos de referencia",
                                 variable=self.show_landmarks).pack(pady=5, anchor="w", padx=20)
        
        ttk.Label(settings_window, text="Umbral de confianza:").pack(pady=(20, 5), anchor="w", padx=20)
        ttk.Scale(settings_window, from_=0.1, to=1.0, variable=self.confidence_threshold,
                                 orient="horizontal").pack(pady=5, padx=20, fill="x")
        
        ttk.Button(settings_window, text="Guardar", command=settings_window.destroy).pack(pady=20)
   
    def show_help(self):
        help_text = """
        SignTranslate - Traductor de Lenguaje de Señas Español
        
        INSTRUCCIONES:
        1. Conecta una cámara web
        2. Haz clic en 'Iniciar Cámara'
        3. Coloca tu mano dentro del área verde
        4. Realiza la seña de la letra que quieres traducir
        5. La letra detectada aparecerá en pantalla
        
        CONSEJOS:
        - Usa un fondo neutro para mejor precisión
        - Buena iluminación es esencial
        - Mantén la mano estable dentro del área
        - Para mejor resultados, recolecta datos primero
        
        Para recolectar datos de entrenamiento:
        1. Haz clic en 'Recolectar Datos'
        2. Selecciona una letra
        3. Realiza la seña y presiona 'c' para capturar
        4. Repite para múltiples ángulos y luces
        """
        messagebox.showinfo("Ayuda - SignTranslate", help_text)