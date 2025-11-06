#!/usr/bin/env python3
"""
Test script for individual components
"""

from src.preprocessing import RetinalImagePreprocessor
from src.vessel_segmentation import VesselSegmenter
from src.feature_extraction import FeatureExtractor
from src.ml_model import HypertensionRiskPredictor
import cv2
import os

def test_preprocessing():
    """Test preprocessing component"""
    print("🔧 Testing Preprocessing Component...")
    try:
        preprocessor = RetinalImagePreprocessor()
        image_path = 'dataset/train/retinal_demo.png'
        
        result = preprocessor.preprocess_pipeline(image_path, save_steps=False)
        print('✅ Preprocessing component working!')
        print(f'   - Original image shape: {result["original"].shape}')
        print(f'   - Preprocessed image shape: {result["preprocessed"].shape}')
        print(f'   - Processing steps completed: {len(result)}')
        return True
    except Exception as e:
        print(f'❌ Preprocessing failed: {e}')
        return False

def test_vessel_segmentation():
    """Test vessel segmentation component"""
    print("\n🩸 Testing Vessel Segmentation Component...")
    try:
        segmenter = VesselSegmenter()
        image_path = 'dataset/train/retinal_demo.png'
        
        # Create temp directory for testing
        os.makedirs('temp_test', exist_ok=True)
        
        result = segmenter.segment_vessels(
            image_path, 
            method='morphological', 
            save_results=False
        )
        print('✅ Vessel segmentation component working!')
        print(f'   - Segmentation mask shape: {result["vessel_mask"].shape}')
        print(f'   - Method used: morphological')
        return True
    except Exception as e:
        print(f'❌ Vessel segmentation failed: {e}')
        return False

def test_feature_extraction():
    """Test feature extraction component"""
    print("\n📊 Testing Feature Extraction Component...")
    try:
        extractor = FeatureExtractor()
        image_path = 'dataset/train/retinal_demo.png'
        
        result = extractor.extract_features(image_path, save_results=False)
        print('✅ Feature extraction component working!')
        print(f'   - AVR (Arteriovenous Ratio): {result["avr"]:.3f}')
        print(f'   - Tortuosity: {result["tortuosity"]:.3f}')
        print(f'   - CDR (Cup-to-Disc Ratio): {result["cdr"]:.3f}')
        return True
    except Exception as e:
        print(f'❌ Feature extraction failed: {e}')
        return False

def test_ml_model():
    """Test ML model component"""
    print("\n🤖 Testing ML Model Component...")
    try:
        predictor = HypertensionRiskPredictor()
        
        # Generate some test data and train a model
        dataset = predictor.generate_synthetic_dataset(n_samples=100)
        X, y = predictor.prepare_data(dataset)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trained_models = predictor.train_models(X_train, y_train)
        print('✅ ML model component working!')
        print(f'   - Models trained: {len(trained_models)}')
        print(f'   - Best model: {type(predictor.best_model).__name__}')
        
        # Test prediction
        sample_features = {'avr': 0.8, 'tortuosity': 1.1, 'cdr': 0.3}
        prediction = predictor.predict_risk(sample_features)
        print(f'   - Sample prediction: {prediction["prediction"]}')
        return True
    except Exception as e:
        print(f'❌ ML model failed: {e}')
        return False

def main():
    """Run all component tests"""
    print("🧪 Testing Individual Components\n")
    print("=" * 50)
    
    results = []
    results.append(test_preprocessing())
    results.append(test_vessel_segmentation())
    results.append(test_feature_extraction())
    results.append(test_ml_model())
    
    print("\n" + "=" * 50)
    print("📋 Component Test Summary")
    print("=" * 50)
    
    components = ['Preprocessing', 'Vessel Segmentation', 'Feature Extraction', 'ML Model']
    
    for i, (component, result) in enumerate(zip(components, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {component}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Cleanup
    if os.path.exists('temp_test'):
        import shutil
        shutil.rmtree('temp_test')

if __name__ == "__main__":
    main()
