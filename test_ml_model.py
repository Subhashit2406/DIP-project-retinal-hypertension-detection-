from src.ml_model import HypertensionRiskPredictor
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(__file__))

# Test ML model
try:
    predictor = HypertensionRiskPredictor()
    predictor.load_model("models/hypertension_risk_predictor.pkl")
    print("Model loaded successfully")

    # Test prediction with sample features
    features = {
        'avr': 0.75,
        'tortuosity': 1.15,
        'cdr': 0.45,
        'vessel_density': 0.85,
        'optic_disc_ratio': 0.25
    }

    try:
        results = predictor.predict_combined_risks(features)
        print("Prediction successful:")
        print(f"Hypertension: {results['hypertension']['prediction']}")
        print(f"Heart Disease: {results['heart_disease']['prediction']}")
        print(f"Overall: {results['overall_risk']['risk_level']}")
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Model loading error: {e}")
    import traceback
    traceback.print_exc()
