from src.preprocessing import RetinalImagePreprocessor
from src.vessel_segmentation import VesselSegmenter
from src.feature_extraction import FeatureExtractor
from src.ml_model import HypertensionRiskPredictor
import cv2
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(__file__))

# Initialize components
try:
    preprocessor = RetinalImagePreprocessor()
    segmenter = VesselSegmenter()
    extractor = FeatureExtractor()
    predictor = HypertensionRiskPredictor()
    predictor.load_model("models/hypertension_risk_predictor.pkl")
    print("All components initialized successfully")
except Exception as e:
    print(f"Initialization error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with the demo image
image_path = 'dataset/train/retinal_demo.png'

try:
    print(f"Processing image: {image_path}")

    # Validate image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not readable")
        exit(1)

    print("Image loaded successfully")

    # Extract features
    features = extractor.extract_features(image_path, save_results=False)
    print(f"Features: AVR={features['avr']}, Tortuosity={features['tortuosity']}, CDR={features['cdr']}")

    # Make prediction
    combined_results = predictor.predict_combined_risks({
        'avr': features['avr'],
        'tortuosity': features['tortuosity'],
        'cdr': features['cdr'],
        'vessel_density': 0.85,
        'optic_disc_ratio': 0.25
    })

    print("Prediction successful:")
    print(f"Hypertension: {combined_results['hypertension']['prediction']}")
    print(f"Heart Disease: {combined_results['heart_disease']['prediction']}")
    print(f"Overall: {combined_results['overall_risk']['risk_level']}")

except Exception as e:
    print(f"Analysis error: {e}")
    import traceback
    traceback.print_exc()
