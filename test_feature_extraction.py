from src.feature_extraction import FeatureExtractor
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(__file__))

# Test feature extraction
try:
    extractor = FeatureExtractor()
    features = extractor.extract_features('dataset/train/retinal_demo.png', save_results=False)
    print('Features extracted successfully:')
    print(f'AVR: {features["avr"]}')
    print(f'Tortuosity: {features["tortuosity"]}')
    print(f'CDR: {features["cdr"]}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
