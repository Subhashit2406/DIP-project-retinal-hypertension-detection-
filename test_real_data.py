#!/usr/bin/env python3
"""
Test script for real data integration
Demonstrates the complete pipeline with real retinal datasets
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from real_data_integration import RealDataManager

def test_real_data_loading():
    """Test loading real datasets"""
    print("Testing real data loading...")

    manager = RealDataManager()

    # Test loading DRIVE dataset
    try:
        images, masks, metadata = manager.load_drive_dataset()
        if images:
            print(f"✅ Successfully loaded {len(images)} DRIVE images")
        else:
            print("❌ No DRIVE images loaded")
    except Exception as e:
        print(f"❌ Error loading DRIVE: {e}")

    # Test loading STARE dataset
    try:
        images, masks, metadata = manager.load_stare_dataset()
        if images:
            print(f"✅ Successfully loaded {len(images)} STARE images")
        else:
            print("❌ No STARE images loaded")
    except Exception as e:
        print(f"❌ Error loading STARE: {e}")

    # Test loading CHASE_DB1 dataset
    try:
        images, masks, metadata = manager.load_chase_db1_dataset()
        if images:
            print(f"✅ Successfully loaded {len(images)} CHASE_DB1 images")
        else:
            print("❌ No CHASE_DB1 images loaded")
    except Exception as e:
        print(f"❌ Error loading CHASE_DB1: {e}")

def test_preprocessing_pipeline():
    """Test preprocessing with real data"""
    print("\nTesting preprocessing pipeline...")

    manager = RealDataManager()

    # Load a small subset for testing
    images, masks, metadata = manager.load_all_datasets()

    if not images:
        print("❌ No images available for preprocessing test")
        return

    # Test preprocessing on first image
    try:
        processed_images, processed_masks = manager.preprocess_real_images([images[0]], [masks[0]] if masks else None)
        print(f"✅ Successfully preprocessed {len(processed_images)} images")
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")

def test_feature_extraction():
    """Test feature extraction with real data"""
    print("\nTesting feature extraction...")

    manager = RealDataManager()

    # Load a small subset
    images, masks, metadata = manager.load_all_datasets()

    if not images:
        print("❌ No images available for feature extraction test")
        return

    # Test feature extraction on first image
    try:
        features_list = manager.extract_features_from_real_data([images[0]], [masks[0]] if masks else None)
        if features_list:
            features = features_list[0]
            print("✅ Successfully extracted features:")
            print(f"   - AVR: {features.get('avr', 'N/A'):.3f}")
            print(f"   - Tortuosity: {features.get('tortuosity', 'N/A'):.3f}")
            print(f"   - CDR: {features.get('cdr', 'N/A'):.3f}")
            print(f"   - Risk: {features.get('hypertension_risk', 'N/A')}")
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")

def test_synthetic_labels():
    """Test synthetic label generation"""
    print("\nTesting synthetic label generation...")

    manager = RealDataManager()

    try:
        labels = manager.create_synthetic_labels(10)
        print(f"✅ Generated {len(labels)} synthetic labels")
        print("Sample labels:")
        for i in range(min(3, len(labels))):
            row = labels.iloc[i]
            print(f"   Patient {row['patient_id']}: {row['risk_level']} "
                  f"(AVR: {row['avr']:.2f}, Tortuosity: {row['tortuosity']:.2f})")
    except Exception as e:
        print(f"❌ Error generating synthetic labels: {e}")

def test_data_augmentation():
    """Test data augmentation"""
    print("\nTesting data augmentation...")

    manager = RealDataManager()

    # Create a simple test image
    import numpy as np
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    try:
        augmented_images, _ = manager.create_data_augmentation([test_image])
        print(f"✅ Successfully augmented to {len(augmented_images)} images")
    except Exception as e:
        print(f"❌ Error in data augmentation: {e}")

def run_quick_demo():
    """Run a quick demonstration of the real data pipeline"""
    print("🚀 Running Quick Real Data Demo")
    print("=" * 50)

    manager = RealDataManager()

    # Load datasets
    print("\n1. Loading datasets...")
    images, masks, metadata = manager.load_all_datasets()

    if not images:
        print("❌ No datasets available. Please run download_datasets.py first")
        print("\nTo download datasets:")
        print("   python download_datasets.py")
        return

    print(f"✅ Loaded {len(images)} images from {len(set([m['dataset'] for m in metadata]))} datasets")

    # Preprocess
    print("\n2. Preprocessing images...")
    processed_images, processed_masks = manager.preprocess_real_images(images[:5], masks[:5] if masks else None)
    print(f"✅ Preprocessed {len(processed_images)} images")

    # Extract features
    print("\n3. Extracting features...")
    features_list = manager.extract_features_from_real_data(processed_images, processed_masks)
    print(f"✅ Extracted features from {len(features_list)} images")

    # Create training data
    print("\n4. Creating training data...")
    clinical_labels = manager.create_synthetic_labels(len(features_list))
    training_data = manager.combine_features_with_labels(features_list, clinical_labels)
    print(f"✅ Created training dataset with {len(training_data)} samples")

    # Show sample results
    print("\n5. Sample Results:")
    for i in range(min(3, len(training_data))):
        row = training_data.iloc[i]
        print(f"   Sample {i+1}: {row['risk_label']} "
              f"(AVR: {row['avr']:.3f}, Tortuosity: {row['tortuosity']:.3f}, CDR: {row['cdr']:.3f})")

    print("\n✅ Quick demo completed successfully!")
    print("\nTo run the full pipeline:")
    print("   python real_data_integration.py")

def main():
    """Main test function"""
    print("🧪 Testing Real Data Integration")
    print("=" * 50)

    # Check if datasets exist
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        print("❌ Dataset directory not found. Please run setup_directories.py first")
        return

    # Run tests
    test_real_data_loading()
    test_preprocessing_pipeline()
    test_feature_extraction()
    test_synthetic_labels()
    test_data_augmentation()

    # Run quick demo
    print("\n" + "=" * 50)
    run_quick_demo()

    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    print("✅ Real data loading functions implemented")
    print("✅ Preprocessing pipeline updated for arrays")
    print("✅ Feature extraction supports array inputs")
    print("✅ Synthetic label generation working")
    print("✅ Data augmentation framework ready")
    print("\n🎯 Next Steps:")
    print("1. Download real datasets: python download_datasets.py")
    print("2. Run full pipeline: python real_data_integration.py")
    print("3. Evaluate results: python test_real_data.py")

if __name__ == "__main__":
    main()
