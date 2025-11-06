#!/usr/bin/env python3
"""
Simple Demo Script to Show What Your DIP Project Can Do
"""

import os
import cv2
from pathlib import Path

def show_project_structure():
    """Show what files and folders exist"""
    print("📁 YOUR DIP PROJECT STRUCTURE:")
    print("=" * 50)

    # Show main directories
    dirs_to_check = ['dataset', 'src', 'models', 'results', 'demo_results']
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
            if dir_name == 'dataset':
                show_dataset_contents()
        else:
            print(f"❌ {dir_name}/ (missing)")

    print("\n📄 MAIN FILES:")
    files_to_check = ['main.py', 'README.md', 'requirements.txt']
    for file_name in files_to_check:
        if os.path.exists(file_name):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} (missing)")

def show_dataset_contents():
    """Show what's in the dataset folder"""
    train_dir = Path('dataset/train')
    if train_dir.exists():
        files = list(train_dir.glob('*'))
        print(f"   ├── train/ ({len(files)} files)")
        for file in files[:3]:  # Show first 3 files
            print(f"   │   ├── {file.name}")
        if len(files) > 3:
            print(f"   │   └── ... and {len(files)-3} more files")

def demonstrate_preprocessing():
    """Show preprocessing working"""
    print("\n🔧 TESTING PREPROCESSING:")
    print("-" * 30)

    try:
        from src.preprocessing import RetinalImagePreprocessor

        preprocessor = RetinalImagePreprocessor()
        print("✅ Preprocessing module loaded")

        # Test with synthetic image
        image_path = "dataset/train/sample_synthetic.png"
        if os.path.exists(image_path):
            results = preprocessor.preprocess_pipeline(image_path, save_steps=False)
            print("✅ Preprocessing pipeline executed successfully")
            print(f"   📊 Generated {len(results)} preprocessing steps")
            return True
        else:
            print("❌ Test image not found")
            return False

    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

def demonstrate_segmentation():
    """Show vessel segmentation working"""
    print("\n🩸 TESTING VESSEL SEGMENTATION:")
    print("-" * 35)

    try:
        from src.vessel_segmentation import VesselSegmenter

        segmenter = VesselSegmenter()
        print("✅ Vessel segmentation module loaded")

        # Test segmentation
        image_path = "dataset/train/sample_synthetic.png"
        if os.path.exists(image_path):
            results = segmenter.segment_vessels(image_path, method='hybrid', save_results=False)
            print("✅ Vessel segmentation executed successfully")
            print("   📊 Generated vessel mask and overlay")
            return True
        else:
            print("❌ Test image not found")
            return False

    except Exception as e:
        print(f"❌ Segmentation error: {e}")
        return False

def demonstrate_feature_extraction():
    """Show feature extraction working"""
    print("\n📊 TESTING FEATURE EXTRACTION:")
    print("-" * 32)

    try:
        from src.feature_extraction import FeatureExtractor

        extractor = FeatureExtractor()
        print("✅ Feature extraction module loaded")

        # Test feature extraction
        image_path = "dataset/train/sample_synthetic.png"
        if os.path.exists(image_path):
            features = extractor.extract_features(image_path, save_results=False)
            print("✅ Feature extraction executed successfully")
            print("   📊 Extracted medical features:")
            print(f"   • AVR: {features['avr']:.3f}")
            print(f"   • Tortuosity: {features['tortuosity']:.3f}")
            print(f"   • CDR: {features['cdr']:.3f}")
            print(f"   🎯 Risk Assessment: {features['hypertension_risk']}")
            return True
        else:
            print("❌ Test image not found")
            return False

    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False

def demonstrate_ml_model():
    """Show ML model working"""
    print("\n🤖 TESTING ML MODEL:")
    print("-" * 20)

    try:
        from src.ml_model import HypertensionRiskPredictor

        predictor = HypertensionRiskPredictor()

        # Try to load trained model
        model_path = "models/hypertension_risk_predictor.pkl"
        if os.path.exists(model_path):
            predictor.load_model(model_path)
            print("✅ ML model loaded from saved file")
        else:
            print("❌ No saved model found - training new model...")
            # Generate synthetic dataset and train
            dataset = predictor.generate_synthetic_dataset(n_samples=1000)
            X, y = predictor.prepare_data(dataset)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            predictor.train_models(X_train, y_train)
            predictor.save_model(model_path)
            print("✅ New model trained and saved")

        # Test prediction
        test_features = {
            'avr': 0.8,
            'tortuosity': 1.3,
            'cdr': 0.4
        }

        prediction = predictor.predict_risk(test_features)
        print("✅ ML prediction executed successfully")
        print("   📊 Test prediction results:")
        print(f"   🎯 Risk Level: {prediction['prediction']}")
        print("   📈 Probabilities:")
        for risk_level, prob in prediction['probabilities'].items():
            print(f"   • {risk_level}: {prob:.1f}")
        return True

    except Exception as e:
        print(f"❌ ML model error: {e}")
        return False

def show_usage_examples():
    """Show how to use the system"""
    print("\n🚀 HOW TO USE YOUR SYSTEM:")
    print("=" * 30)

    print("1️⃣ ANALYZE A SINGLE IMAGE:")
    print("   python main.py --image dataset/train/sample_synthetic.png")
    print()

    print("2️⃣ PROCESS MULTIPLE IMAGES:")
    print("   python main.py --batch dataset/train/")
    print()

    print("3️⃣ TRAIN A NEW MODEL:")
    print("   python main.py --train")
    print()

    print("4️⃣ USE WITH YOUR OWN IMAGES:")
    print("   python main.py --image path/to/your/retinal_image.png")
    print()

def main():
    """Run the complete demo"""
    print("🎯 HYPERTENSION DETECTION DIP PROJECT - DEMO")
    print("=" * 55)
    print("This demo shows what your complete system can do!")
    print()

    # Show project structure
    show_project_structure()
    print()

    # Test each component
    preprocessing_ok = demonstrate_preprocessing()
    segmentation_ok = demonstrate_segmentation()
    feature_ok = demonstrate_feature_extraction()
    ml_ok = demonstrate_ml_model()

    print("\n📋 COMPONENT STATUS SUMMARY:")
    print("-" * 30)
    print(f"🔧 Preprocessing: {'✅ WORKING' if preprocessing_ok else '❌ FAILED'}")
    print(f"🩸 Segmentation: {'✅ WORKING' if segmentation_ok else '❌ FAILED'}")
    print(f"📊 Feature Extraction: {'✅ WORKING' if feature_ok else '❌ FAILED'}")
    print(f"🤖 ML Model: {'✅ WORKING' if ml_ok else '❌ FAILED'}")

    working_components = sum([preprocessing_ok, segmentation_ok, feature_ok, ml_ok])
    print(f"\n🎯 SYSTEM STATUS: {working_components}/4 components working")

    if working_components == 4:
        print("🎉 YOUR DIP PROJECT IS FULLY FUNCTIONAL!")
    else:
        print("⚠️ Some components need fixing")

    print()
    show_usage_examples()

    print("\n💡 WHAT YOUR PROJECT DOES:")
    print("-" * 30)
    print("• Analyzes retinal fundus images")
    print("• Detects blood vessels")
    print("• Extracts medical features (AVR, tortuosity, CDR)")
    print("• Predicts hypertension risk")
    print("• Generates professional medical reports")
    print("• Creates visualizations")

    print("\n🎯 READY TO USE WITH ANY RETINAL IMAGES!")

if __name__ == "__main__":
    main()
