"""
Script to train the initial model
Run this if you want to train the model from command line instead of using the notebook
"""
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

from preprocessing import prepare_dataset, augment_data
from model import create_model, train_model, evaluate_model, save_model_metadata

def main():
    print("="*60)
    print("Cat and Dog Classification Model Training")
    print("="*60)
    
    # Define paths
    train_cats_dir = 'data/train/cats'
    train_dogs_dir = 'data/train/dogs'
    test_cats_dir = 'data/test/cats'
    test_dogs_dir = 'data/test/dogs'
    model_path = 'models/cat_dog_model.h5'
    
    # Check if data directories exist
    for dir_path in [train_cats_dir, train_dogs_dir, test_cats_dir, test_dogs_dir]:
        if not os.path.exists(dir_path):
            print(f"Error: Data directory not found: {dir_path}")
            print("Please download the dataset and place it in the data/ directory")
            return
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    X_train, X_test, y_train, y_test = prepare_dataset(
        train_cats_dir, train_dogs_dir,
        test_cats_dir, test_dogs_dir,
        target_size=(224, 224),
        max_train=1000,  # Use subset for faster training
        max_test=200
    )
    
    # Augment data
    print("\n[2/5] Augmenting training data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, augment_factor=1)
    
    # Split into train and validation
    print("\n[3/5] Splitting data...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_aug, y_train_aug,
        test_size=0.2,
        random_state=42,
        stratify=y_train_aug
    )
    
    # Create model
    print("\n[4/5] Creating model with transfer learning...")
    model = create_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        use_pretrained=True
    )
    
    # Train model
    print("\n[5/5] Training model...")
    print("This may take a while. Please be patient...")
    history = train_model(
        model, X_train_final, y_train_final, X_val, y_val,
        epochs=20,
        batch_size=32,
        model_save_path=model_path
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metadata
    save_model_metadata(metrics, 'models/model_metadata.json')
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print("You can now run the Flask app: python app.py")

if __name__ == '__main__':
    main()

