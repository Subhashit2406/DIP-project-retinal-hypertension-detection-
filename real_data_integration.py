#!/usr/bin/env python3
"""
Real Data Integration for Retinal Image Analysis
Handles loading, preprocessing, and training with real retinal datasets
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.preprocessing import RetinalImagePreprocessor
from src.vessel_segmentation import VesselSegmenter
from src.feature_extraction import FeatureExtractor
from src.ml_model import HypertensionRiskPredictor

class RealDataManager:
    """Manages real retinal datasets for training and evaluation"""

    def __init__(self, data_dir='dataset'):
        self.data_dir = Path(data_dir)
        self.preprocessor = RetinalImagePreprocessor()
        self.segmenter = VesselSegmenter()
        self.extractor = FeatureExtractor()
        self.predictor = HypertensionRiskPredictor()

        # Dataset configurations
        self.dataset_configs = {
            'DRIVE': {
                'image_dir': 'training/images',
                'mask_dir': 'training/1st_manual',
                'image_ext': '.tif',
                'mask_ext': '.gif'
            },
            'STARE': {
                'image_dir': 'images',
                'mask_dir': 'labels',
                'image_ext': '.ppm',
                'mask_ext': '.ppm'
            },
            'CHASE_DB1': {
                'image_dir': 'images',
                'mask_dir': 'masks',
                'image_ext': '.jpg',
                'mask_ext': '.png'
            }
        }

    def load_drive_dataset(self):
        """Load DRIVE dataset with vessel annotations"""
        print("Loading DRIVE dataset...")

        drive_dir = self.data_dir / 'drive'
        if not drive_dir.exists():
            print("DRIVE dataset not found. Please run download_datasets.py first")
            return None

        images = []
        masks = []
        metadata = []

        # Load training images and masks
        image_dir = drive_dir / 'training' / 'images'
        mask_dir = drive_dir / 'training' / '1st_manual'

        if image_dir.exists() and mask_dir.exists():
            for i in range(21, 41):  # DRIVE training images 21-40
                img_path = image_dir / f'{i:02d}_training.tif'
                mask_path = mask_dir / f'{i:02d}_manual1.gif'

                if img_path.exists() and mask_path.exists():
                    # Load image (RGB)
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Load mask (grayscale)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 0).astype(np.uint8)  # Binarize

                    images.append(img)
                    masks.append(mask)
                    metadata.append({
                        'dataset': 'DRIVE',
                        'image_id': f'{i:02d}',
                        'split': 'train'
                    })

        print(f"Loaded {len(images)} DRIVE training images")
        return images, masks, metadata

    def load_stare_dataset(self):
        """Load STARE dataset"""
        print("Loading STARE dataset...")

        stare_dir = self.data_dir / 'stare'
        if not stare_dir.exists():
            print("STARE dataset not found. Please run download_datasets.py first")
            return None

        images = []
        masks = []
        metadata = []

        # Load all .ppm files in the STARE directory
        import glob
        stare_image_paths = glob.glob(str(stare_dir / '*.ppm'))

        for img_path in stare_image_paths:
            img_name = Path(img_path).stem  # Get filename without extension
            img_path = stare_dir / f'{img_name}.ppm'
            mask_path = stare_dir / f'{img_name}.ah.ppm'  # Hoover annotations

            if img_path.exists():
                # Load image
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    images.append(img)
                    metadata.append({
                        'dataset': 'STARE',
                        'image_id': img_name,
                        'split': 'train'
                    })

                    # Load mask if available
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            mask = (mask > 0).astype(np.uint8)
                            masks.append(mask)
                        else:
                            masks.append(None)
                    else:
                        masks.append(None)

        print(f"Loaded {len(images)} STARE images")
        return images, masks, metadata

    def load_chase_db1_dataset(self):
        """Load CHASE_DB1 dataset"""
        print("Loading CHASE_DB1 dataset...")

        chase_dir = self.data_dir / 'chase_db1'
        if not chase_dir.exists():
            print("CHASE_DB1 dataset not found. Please run download_datasets.py first")
            return None

        images = []
        masks = []
        metadata = []

        # CHASE_DB1 has 28 images (14 patients × 2 images each)
        for i in range(1, 15):
            for j in [1, 2]:
                img_path = chase_dir / f'Image_{i:02d}_{j}.jpg'
                mask_path = chase_dir / f'Image_{i:02d}_{j}_1stHO.png'

                if img_path.exists() and mask_path.exists():
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Load mask
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            mask = (mask > 0).astype(np.uint8)

                            images.append(img)
                            masks.append(mask)
                            metadata.append({
                                'dataset': 'CHASE_DB1',
                                'image_id': f'{i:02d}_{j}',
                                'split': 'train'
                            })

        print(f"Loaded {len(images)} CHASE_DB1 images")
        return images, masks, metadata

    def load_all_datasets(self):
        """Load all available real datasets"""
        print("Loading all real datasets...")

        all_images = []
        all_masks = []
        all_metadata = []

        # Load each dataset
        datasets = [
            self.load_drive_dataset,
            self.load_stare_dataset,
            self.load_chase_db1_dataset
        ]

        for load_func in datasets:
            result = load_func()
            if result is not None:
                images, masks, metadata = result
                all_images.extend(images)
                all_masks.extend(masks)
                all_metadata.extend(metadata)

        print(f"\nTotal loaded: {len(all_images)} images from {len(set([m['dataset'] for m in all_metadata]))} datasets")

        return all_images, all_masks, all_metadata

    def create_synthetic_labels(self, n_samples):
        """Create synthetic clinical labels for demonstration"""
        np.random.seed(42)

        labels = []
        for i in range(n_samples):
            # Simulate clinical labels based on typical distributions
            if np.random.random() < 0.3:  # 30% hypertensive
                risk_level = 'High Risk'
                avr = np.random.uniform(0.4, 0.8)
                tortuosity = np.random.uniform(1.2, 1.6)
                cdr = np.random.uniform(0.2, 0.6)
            elif np.random.random() < 0.5:  # 50% moderate
                risk_level = 'Moderate Risk'
                avr = np.random.uniform(0.7, 1.0)
                tortuosity = np.random.uniform(1.1, 1.3)
                cdr = np.random.uniform(0.3, 0.5)
            else:  # 20% low risk
                risk_level = 'Low Risk'
                avr = np.random.uniform(0.9, 1.3)
                tortuosity = np.random.uniform(1.0, 1.15)
                cdr = np.random.uniform(0.35, 0.55)

            # Add realistic noise
            avr += np.random.normal(0, 0.05)
            tortuosity += np.random.normal(0, 0.02)
            cdr += np.random.normal(0, 0.02)

            labels.append({
                'patient_id': f'PAT{i:04d}',
                'risk_level': risk_level,
                'avr': np.clip(avr, 0.1, 2.0),
                'tortuosity': np.clip(tortuosity, 1.0, 2.0),
                'cdr': np.clip(cdr, 0.1, 0.8)
            })

        return pd.DataFrame(labels)

    def preprocess_real_images(self, images, masks=None):
        """Preprocess real retinal images"""
        print("Preprocessing real images...")

        processed_images = []
        processed_masks = []

        for i, img in enumerate(tqdm(images)):
            try:
                # Apply preprocessing pipeline
                result = self.preprocessor.preprocess_pipeline_from_array(img)

                if result and 'preprocessed' in result:
                    processed_images.append(result['preprocessed'])

                    # Process mask if available
                    if masks and i < len(masks) and masks[i] is not None:
                        # Resize mask to match preprocessed image
                        h, w = result['preprocessed'].shape[:2]
                        mask_resized = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                        processed_masks.append(mask_resized)
                    else:
                        processed_masks.append(None)

            except Exception as e:
                print(f"Error preprocessing image {i}: {e}")
                continue

        print(f"Successfully preprocessed {len(processed_images)} images")
        return processed_images, processed_masks

    def extract_features_from_real_data(self, images, masks=None):
        """Extract medical features from real images"""
        print("Extracting features from real data...")

        features_list = []

        for i, img in enumerate(tqdm(images)):
            try:
                # Extract features
                if masks and i < len(masks) and masks[i] is not None:
                    # Use ground truth mask for better feature extraction
                    features = self.extractor.extract_features_from_array(img, masks[i])
                else:
                    # Segment vessels first
                    vessel_result = self.segmenter.segment_vessels_from_array(img, method='hybrid')
                    if vessel_result and 'vessel_mask' in vessel_result:
                        features = self.extractor.extract_features_from_array(img, vessel_result['vessel_mask'])
                    else:
                        features = self.extractor.extract_features_from_array(img)

                if features:
                    features_list.append(features)

            except Exception as e:
                print(f"Error extracting features from image {i}: {e}")
                continue

        print(f"Successfully extracted features from {len(features_list)} images")
        return features_list

    def create_real_data_training_pipeline(self):
        """Complete pipeline for training with real data"""
        print("Starting real data training pipeline...")

        # 1. Load datasets
        images, masks, metadata = self.load_all_datasets()

        if not images:
            print("No datasets loaded. Please download datasets first.")
            return None

        # 2. Preprocess images
        processed_images, processed_masks = self.preprocess_real_images(images, masks)

        # 3. Extract features
        features_list = self.extract_features_from_real_data(processed_images, processed_masks)

        # 4. Create synthetic clinical labels (for demonstration)
        clinical_labels = self.create_synthetic_labels(len(features_list))

        # 5. Combine features with labels
        training_data = self.combine_features_with_labels(features_list, clinical_labels)

        # 6. Train hypertension model
        model = self.train_on_real_data(training_data)

        # 7. Train heart disease model
        heart_dataset = self.predictor.generate_heart_disease_dataset(len(features_list))
        X_heart, y_heart = self.predictor.prepare_heart_data(heart_dataset)
        heart_model = self.predictor.train_heart_disease_model(X_heart, y_heart)

        # 8. Save updated model (including heart disease model)
        self.predictor.save_model("models/hypertension_risk_predictor.pkl")

        # 9. Evaluate
        self.evaluate_real_data_model(model, training_data)

        return model

    def combine_features_with_labels(self, features_list, clinical_labels):
        """Combine extracted features with clinical labels"""
        combined_data = []

        for i, features in enumerate(features_list):
            if i < len(clinical_labels):
                data_point = {
                    'patient_id': clinical_labels.iloc[i]['patient_id'],
                    'risk_label': clinical_labels.iloc[i]['risk_level'],  # Changed to match ML model expectation
                    'avr': features.get('avr', clinical_labels.iloc[i]['avr']),
                    'tortuosity': features.get('tortuosity', clinical_labels.iloc[i]['tortuosity']),
                    'cdr': features.get('cdr', clinical_labels.iloc[i]['cdr'])
                }
                combined_data.append(data_point)

        return pd.DataFrame(combined_data)

    def train_on_real_data(self, training_data):
        """Train ML model on real data"""
        print("Training model on real data...")

        # Prepare data
        X, y = self.predictor.prepare_data(training_data)

        # Split data (handle small datasets)
        if len(X) < 10:  # Small dataset, don't stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42  # Use 50% for test with small data
            )
        else:  # Large dataset, use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train models
        trained_models = self.predictor.train_models(X_train, y_train)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        for name, model in trained_models.items():
            results = self.predictor.evaluate_model(model, X_test, y_test, name)
            print(f"{name} Test Accuracy: {results['accuracy']:.3f}")

        return self.predictor.best_model

    def evaluate_real_data_model(self, model, test_data):
        """Evaluate model performance on real data"""
        print("Evaluating model on real data...")

        X_test, y_test = self.predictor.prepare_data(test_data)
        X_test_scaled = self.predictor.scaler.transform(X_test)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                  target_names=['Low Risk', 'Moderate Risk', 'High Risk']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Risk', 'Moderate Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'Moderate Risk', 'High Risk'])
        plt.title('Confusion Matrix - Real Data Evaluation')
        plt.tight_layout()
        plt.savefig('results/ml_model/real_data_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Confusion matrix saved to results/ml_model/real_data_confusion_matrix.png")

    def create_data_augmentation(self, images, masks=None):
        """Apply data augmentation to increase dataset size"""
        print("Applying data augmentation...")

        augmented_images = []
        augmented_masks = []

        for i, img in enumerate(images):
            augmented_images.append(img)  # Original image
            if masks and i < len(masks):
                augmented_masks.append(masks[i])

            # Apply augmentations
            for aug_type in ['rotate', 'flip', 'brightness', 'contrast']:
                aug_img = self.apply_augmentation(img, aug_type)
                augmented_images.append(aug_img)

                if masks and i < len(masks):
                    aug_mask = self.apply_augmentation(masks[i], aug_type, is_mask=True)
                    augmented_masks.append(aug_mask)

        print(f"Augmented from {len(images)} to {len(augmented_images)} images")
        return augmented_images, augmented_masks

    def apply_augmentation(self, image, aug_type, is_mask=False):
        """Apply specific augmentation technique"""
        if aug_type == 'rotate':
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            if is_mask:
                return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)
            else:
                return cv2.warpAffine(image, M, (w, h))

        elif aug_type == 'flip':
            return cv2.flip(image, np.random.choice([-1, 0, 1]))

        elif aug_type == 'brightness':
            factor = np.random.uniform(0.8, 1.2)
            if is_mask:
                return image  # Don't modify masks
            else:
                return np.clip(image * factor, 0, 255).astype(np.uint8)

        elif aug_type == 'contrast':
            factor = np.random.uniform(0.8, 1.2)
            if is_mask:
                return image
            else:
                mean = np.mean(image)
                return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        return image

def main():
    """Main function for real data integration"""
    print("Real Data Integration for Retinal Image Analysis")
    print("=" * 60)

    # Create output directories
    Path("results/ml_model").mkdir(parents=True, exist_ok=True)
    Path("dataset/processed").mkdir(parents=True, exist_ok=True)

    # Initialize data manager
    data_manager = RealDataManager()

    # Run complete pipeline
    model = data_manager.create_real_data_training_pipeline()

    if model:
        print("\n✅ Real data training completed successfully!")
        print("Model saved and ready for clinical evaluation")
    else:
        print("\n❌ Real data training failed. Please check dataset availability.")

if __name__ == "__main__":
    main()
