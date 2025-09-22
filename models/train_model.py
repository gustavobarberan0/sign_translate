import os
import json
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split

# Importaciones de TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    print("✅ TensorFlow importado correctamente")
except ImportError as e:
    print(f"❌ Error importando TensorFlow: {e}")
    print("Instala con: pip install tensorflow")
    exit()

DATA_PATH = os.path.join('data', 'collected_data')
MODEL_PATH = 'models'

# Crear directorio de modelos si no existe
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    print(f"Directorio {MODEL_PATH} creado")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks_from_image(image_path):
    """Extrae landmarks de una imagen usando MediaPipe"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo leer la imagen {image_path}")
            return None
        
        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen con MediaPipe
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            return np.array(landmarks)
        return None
    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return None

def train_model():
    """Entrena un modelo de IA a partir de los datos recolectados."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: La carpeta {DATA_PATH} no existe. Por favor, recolecta datos primero.")
        return

    data = []
    labels = []
    
    # Obtener clases (letras)
    class_labels = []
    for item in os.listdir(DATA_PATH):
        item_path = os.path.join(DATA_PATH, item)
        if os.path.isdir(item_path):
            class_labels.append(item)
    
    class_labels.sort()
    
    if not class_labels:
        print("No se encontraron clases (letras) en la carpeta de datos.")
        return
    
    print("Iniciando la carga de datos para el entrenamiento...")
    print(f"Clases encontradas: {class_labels}")
    
    # Procesar imágenes
    for i, label in enumerate(class_labels):
        folder_path = os.path.join(DATA_PATH, label)
        print(f"Procesando clase {label} ({i+1}/{len(class_labels)})...")
        
        image_count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(folder_path, filename)
                landmarks = extract_landmarks_from_image(file_path)
                
                if landmarks is not None and len(landmarks) == 63:
                    data.append(landmarks)
                    labels.append(i)
                    image_count += 1
        
        print(f"  -> {image_count} imágenes procesadas para {label}")
    
    if not data:
        print("No se encontraron datos válidos. Asegúrate de:")
        print("1. Tener imágenes con manos visibles")
        print("2. Las imágenes estén en formato JPG o PNG")
        print("3. MediaPipe pueda detectar las manos")
        return

    # Convertir a arrays de numpy
    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"Datos cargados: {X.shape[0]} muestras con {len(class_labels)} clases.")
    
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba: {X_test.shape[0]} muestras")
    
    # Crear el modelo
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(class_labels), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("Resumen del modelo:")
    model.summary()
    
    # Entrenar
    print("Entrenando el modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluar
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisión en prueba: {test_acc:.4f}")
    
    # Guardar modelo
    model_path = os.path.join(MODEL_PATH, 'lse_model.h5')
    model.save(model_path)
    
    # Guardar mapeo de clases
    class_mapping = {i: label for i, label in enumerate(class_labels)}
    mapping_path = os.path.join(MODEL_PATH, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f)
    
    print(f"\n✅ Modelo guardado en: {model_path}")
    print(f"✅ Mapping de clases guardado en: {mapping_path}")
    print(f"✅ Clases: {class_mapping}")

if __name__ == "__main__":
    print("=== ENTRENAMIENTO DEL MODELO SIGNTRANSLATE ===")
    train_model()