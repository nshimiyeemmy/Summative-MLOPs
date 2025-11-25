"""
Model creation and training module
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import json
from datetime import datetime


def create_model(input_shape=(224, 224, 3), num_classes=2, use_pretrained=True):
    """
    Create a CNN model using transfer learning with MobileNetV2
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes (2 for cat/dog)
        use_pretrained: Whether to use pretrained MobileNetV2 weights
    
    Returns:
        Compiled model
    """
    # Base model (pretrained MobileNetV2)
    if use_pretrained:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base model initially
    else:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, 
                model_save_path='models/cat_dog_model.h5'):
    """
    Train the model with early stopping and model checkpointing
    
    Args:
        model: Model to train
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        model_save_path: Path to save the best model
    
    Returns:
        Training history
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return comprehensive metrics
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Loss
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def save_model_metadata(metrics, model_path='models/model_metadata.json'):
    """
    Save model metadata including metrics and training info
    
    Args:
        metrics: Dictionary of metrics
        model_path: Path to save metadata
    """
    metadata = {
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path
    }
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_model(model_path='models/cat_dog_model.h5'):
    """
    Load a saved model
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return keras.models.load_model(model_path)


def retrain_model(X_new, y_new, base_model_path='models/cat_dog_model.h5', 
                 new_model_path='models/cat_dog_model_retrained.h5',
                 epochs=10, batch_size=32):
    """
    Retrain model with new data
    
    Args:
        X_new: New training images
        y_new: New training labels
        base_model_path: Path to base model
        new_model_path: Path to save retrained model
        epochs: Number of epochs for retraining
        batch_size: Batch size
    
    Returns:
        Retrained model and history
    """
    # Load base model
    model = load_model(base_model_path)
    
    # Unfreeze some layers for fine-tuning
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-10]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train on new data
    history = model.fit(
        X_new, y_new,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Save retrained model
    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
    model.save(new_model_path)
    
    return model, history

