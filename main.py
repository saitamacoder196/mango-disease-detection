# File: main.py
# Entry point của chương trình

import argparse
import os
import yaml
from src.data.preprocessing import preprocess_dataset
from src.data.augmentation import augment_dataset
from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_learning import build_transfer_model
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.inference.predict import predict_image

def main():
    parser = argparse.ArgumentParser(description='Mango Disease Detection')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'train', 'evaluate', 'predict'],
                        help='Mode: preprocess, train, evaluate, predict')
    parser.add_argument('--config', type=str, default='configs/custom_cnn_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_type', type=str, default='custom_cnn',
                        choices=['custom_cnn', 'transfer_learning'],
                        help='Type of model to use')
    parser.add_argument('--image_path', type=str,
                        help='Path to image for prediction (only in predict mode)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.mode == 'preprocess':
        print("Preprocessing dataset...")
        preprocess_dataset(
            input_dir=config['data']['raw_dir'],
            output_dir=config['data']['processed_dir'],
            img_size=config['data']['img_size'],
            test_split=config['data']['test_split'],
            validation_split=config['data']['validation_split']
        )
        
        if config['data']['use_augmentation']:
            print("Augmenting dataset...")
            augment_dataset(
                input_dir=os.path.join(config['data']['processed_dir'], 'train'),
                output_dir=config['data']['augmented_dir'],
                augmentation_config=config['augmentation']
            )
    
    elif args.mode == 'train':
        print(f"Training {args.model_type} model...")
        
        # Build model based on selected type
        if args.model_type == 'custom_cnn':
            model = build_custom_cnn(
                input_shape=config['model']['input_shape'],
                num_classes=config['model']['num_classes'],
                model_config=config['model']['custom_cnn']
            )
        else:  # transfer_learning
            model = build_transfer_model(
                input_shape=config['model']['input_shape'],
                num_classes=config['model']['num_classes'],
                base_model=config['model']['transfer_learning']['base_model'],
                model_config=config['model']['transfer_learning']
            )
        
        # Train model
        train_model(
            model=model,
            train_dir=config['data']['train_dir'],
            validation_dir=config['data']['validation_dir'],
            batch_size=config['training']['batch_size'],
            epochs=config['training']['epochs'],
            learning_rate=config['training']['learning_rate'],
            model_save_path=os.path.join(config['model']['save_dir'], f"{args.model_type}_model.h5"),
            training_config=config['training']
        )
    
    elif args.mode == 'evaluate':
        print(f"Evaluating {args.model_type} model...")
        
        evaluate_model(
            model_path=os.path.join(config['model']['save_dir'], f"{args.model_type}_model.h5"),
            test_dir=config['data']['test_dir'],
            batch_size=config['evaluation']['batch_size'],
            img_size=config['model']['input_shape'][:2],
            evaluation_config=config['evaluation']
        )
    
    elif args.mode == 'predict':
        if not args.image_path:
            parser.error("--image_path is required in predict mode")
        
        print(f"Predicting with {args.model_type} model...")
        
        prediction = predict_image(
            model_path=os.path.join(config['model']['save_dir'], f"{args.model_type}_model.h5"),
            image_path=args.image_path,
            img_size=config['model']['input_shape'][:2],
            class_names=config['model']['class_names']
        )
        
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()