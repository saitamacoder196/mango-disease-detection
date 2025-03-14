# train_unet.py - Script simplificado para entrenar U-Net
import os
import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from tqdm import tqdm

# Configuración
CONFIG = {
    'data': {
        'train_dir': 'data/segmentation/train',
        'validation_dir': 'data/segmentation/val',
        'test_dir': 'data/segmentation/test',
        'img_size': [512, 512]
    },
    'model': {
        'input_shape': [512, 512, 3],
        'num_classes': 6,
        'save_dir': 'models'
    },
    'training': {
        'batch_size': 8,
        'epochs': 30,
        'learning_rate': 0.0001,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5
    }
}

# Crear U-Net
def build_unet(input_shape=(512, 512, 3), num_classes=6):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.25)(p1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.25)(p2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.25)(p3)
    
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.25)(p4)
    
    # Bridge
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    c6 = Dropout(0.25)(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    c7 = Dropout(0.25)(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    c8 = Dropout(0.25)(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['training']['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Cargador de datos simple
class SegmentationDataLoader:
    def __init__(self, data_dir, img_size=(512, 512), batch_size=8, num_classes=6):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        
        self.image_paths = sorted([
            os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        self.mask_paths = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(self.masks_dir, f"{base_name}.png")
            if os.path.exists(mask_path):
                self.mask_paths.append(mask_path)
            else:
                # Si no encontramos la máscara, eliminamos la imagen
                self.image_paths.remove(img_path)
        
        self.num_samples = len(self.image_paths)
        self.steps_per_epoch = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.steps_per_epoch += 1
            
        print(f"Encontrado {self.num_samples} imágenes en {data_dir}")
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        batch_size = end_idx - start_idx
        
        # Inicializar arrays para imágenes y máscaras
        batch_imgs = np.zeros((batch_size, *self.img_size, 3), dtype=np.float32)
        batch_masks = np.zeros((batch_size, *self.img_size, self.num_classes), dtype=np.float32)
        
        # Cargar imágenes y máscaras
        for i in range(batch_size):
            idx_in_data = start_idx + i
            
            # Cargar imagen
            img = cv2.imread(self.image_paths[idx_in_data])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0  # Normalizar
            
            # Cargar máscara
            mask = cv2.imread(self.mask_paths[idx_in_data], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            
            # One-hot encoding para máscara
            mask_one_hot = np.zeros((*self.img_size, self.num_classes), dtype=np.float32)
            for class_idx in range(self.num_classes):
                mask_one_hot[:, :, class_idx] = (mask == class_idx).astype(np.float32)
            
            # Agregar a batch
            batch_imgs[i] = img
            batch_masks[i] = mask_one_hot
        
        return batch_imgs, batch_masks
    
    def on_epoch_end(self):
        # Barajar los datos al final de cada época
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.mask_paths = [self.mask_paths[i] for i in indices]

# Función principal de entrenamiento
def train_unet():
    # Crear directorios de salida
    os.makedirs(CONFIG['model']['save_dir'], exist_ok=True)
    
    # Crear generadores de datos
    train_loader = SegmentationDataLoader(
        CONFIG['data']['train_dir'],
        img_size=tuple(CONFIG['data']['img_size']),
        batch_size=CONFIG['training']['batch_size'],
        num_classes=CONFIG['model']['num_classes']
    )
    
    val_loader = SegmentationDataLoader(
        CONFIG['data']['validation_dir'],
        img_size=tuple(CONFIG['data']['img_size']),
        batch_size=CONFIG['training']['batch_size'],
        num_classes=CONFIG['model']['num_classes']
    )
    
    # Crear modelo
    model = build_unet(
        input_shape=tuple(CONFIG['model']['input_shape']),
        num_classes=CONFIG['model']['num_classes']
    )
    
    # Resumen del modelo
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(CONFIG['model']['save_dir'], 'unet_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=CONFIG['training']['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Entrenamiento manual con generadores personalizados
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(CONFIG['training']['epochs']):
        print(f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
        
        # Entrenamiento
        train_loss = 0
        train_accuracy = 0
        
        for batch_idx in tqdm(range(len(train_loader)), desc="Training"):
            batch_imgs, batch_masks = train_loader[batch_idx]
            metrics = model.train_on_batch(batch_imgs, batch_masks)
            train_loss += metrics[0]
            train_accuracy += metrics[1]
        
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        
        # Validación
        val_loss = 0
        val_accuracy = 0
        
        for batch_idx in tqdm(range(len(val_loader)), desc="Validation"):
            batch_imgs, batch_masks = val_loader[batch_idx]
            metrics = model.test_on_batch(batch_imgs, batch_masks)
            val_loss += metrics[0]
            val_accuracy += metrics[1]
        
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        
        print(f"Train loss: {train_loss:.4f} - Train accuracy: {train_accuracy:.4f}")
        print(f"Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f}")
        
        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model.save(os.path.join(CONFIG['model']['save_dir'], 'unet_model.h5'))
            print(f"Modelo guardado - Val loss mejoró a {best_val_loss:.4f}")
        
        # Early stopping
        if epoch - best_epoch >= CONFIG['training']['early_stopping_patience']:
            print(f"Early stopping en epoch {epoch+1}")
            break
        
        # Preparar para la siguiente época
        train_loader.on_epoch_end()
        val_loader.on_epoch_end()
    
    print(f"Entrenamiento completado. Mejor val_loss: {best_val_loss:.4f} en epoch {best_epoch+1}")
    return model

if __name__ == "__main__":
    # Leer configuración de YAML si existe
    if os.path.exists('configs/segmentation_config_new.yaml'):
        with open('configs/segmentation_config_new.yaml', 'r') as f:
            try:
                yaml_config = yaml.safe_load(f)
                # Actualizar la configuración con los valores del YAML
                CONFIG['data'].update(yaml_config.get('data', {}))
                CONFIG['model'].update(yaml_config.get('model', {}))
                CONFIG['training'].update(yaml_config.get('training', {}))
                print("Configuración cargada desde YAML")
            except Exception as e:
                print(f"Error al cargar YAML: {e}")
    
    train_unet()