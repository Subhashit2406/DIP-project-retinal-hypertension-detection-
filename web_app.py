#!/usr/bin/env python3
"""
Hypertension Detection Web Application
A simple web interface for users to upload retinal images and get hypertension risk analysis
"""

from flask import Flask, request, render_template, flash, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import requests
import tempfile

# Import our DIP modules
from src.preprocessing import RetinalImagePreprocessor
from src.vessel_segmentation import VesselSegmenter
from src.feature_extraction import FeatureExtractor
from src.ml_model import HypertensionRiskPredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hypertension_detection_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Initialize our DIP components
preprocessor = RetinalImagePreprocessor()
segmenter = VesselSegmenter()
extractor = FeatureExtractor()

# Try to load trained model
try:
    predictor = HypertensionRiskPredictor()
    predictor.load_model("models/hypertension_risk_predictor.pkl")
    model_loaded = True
except:
    predictor = None
    model_loaded = False

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_visualization(image_path, results, output_path):
    """Create a simple visualization for web display"""
    try:
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            return False

        # Create a simple visualization
        plt.figure(figsize=(12, 8))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Your Retinal Image')
        plt.axis('off')

        # Add feature values as text
        plt.subplot(2, 3, 2)
        plt.text(0.1, 0.8, "ANALYSIS RESULTS", fontsize=14, fontweight='bold')
        plt.text(0.1, 0.6, f"AVR: {results.get('avr', 'N/A'):.3f}", fontsize=12)
        plt.text(0.1, 0.5, f"Tortuosity: {results.get('tortuosity', 'N/A'):.3f}", fontsize=12)
        plt.text(0.1, 0.4, f"CDR: {results.get('cdr', 'N/A'):.3f}", fontsize=12)
        plt.text(0.1, 0.3, f"Risk: {results.get('hypertension_risk', 'Unknown')}", fontsize=12, color='red')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # Risk level visualization
        plt.subplot(2, 3, 3)
        risk_colors = {'Low Risk': 'green', 'Moderate Risk': 'orange', 'High Risk': 'red'}
        risk_color = risk_colors.get(results.get('hypertension_risk', 'Unknown'), 'gray')

        # Create a colored square for risk level
        plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color=risk_color, alpha=0.7)
        plt.text(0.5, 0.5, results.get('hypertension_risk', 'Unknown'),
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Risk Assessment')
        plt.axis('off')

        # Medical interpretation
        plt.subplot(2, 3, 4)
        plt.text(0.1, 0.9, "MEDICAL INTERPRETATION", fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, "• AVR: Ratio of artery to vein diameters", fontsize=10)
        plt.text(0.1, 0.6, "• Tortuosity: Vessel curvature measure", fontsize=10)
        plt.text(0.1, 0.5, "• CDR: Optic cup to disc ratio", fontsize=10)
        plt.text(0.1, 0.3, "⚠️  Consult a doctor for diagnosis", fontsize=10, color='red')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # Recommendations
        plt.subplot(2, 3, 5)
        risk_level = results.get('hypertension_risk', 'Unknown')
        plt.text(0.1, 0.9, "RECOMMENDATIONS", fontsize=12, fontweight='bold')

        if risk_level == 'High Risk':
            plt.text(0.1, 0.7, "• Schedule medical check-up soon", fontsize=10)
            plt.text(0.1, 0.6, "• Monitor blood pressure regularly", fontsize=10)
            plt.text(0.1, 0.5, "• Consider lifestyle changes", fontsize=10)
            plt.text(0.1, 0.4, "• Consult cardiologist", fontsize=10)
        elif risk_level == 'Moderate Risk':
            plt.text(0.1, 0.7, "• Regular health check-ups", fontsize=10)
            plt.text(0.1, 0.6, "• Maintain healthy lifestyle", fontsize=10)
            plt.text(0.1, 0.5, "• Monitor blood pressure", fontsize=10)
        else:
            plt.text(0.1, 0.7, "• Continue healthy habits", fontsize=10)
            plt.text(0.1, 0.6, "• Regular eye exams", fontsize=10)
            plt.text(0.1, 0.5, "• Maintain cardiovascular health", fontsize=10)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # Disclaimer
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, "⚠️ IMPORTANT DISCLAIMER", fontsize=10, fontweight='bold', color='red')
        plt.text(0.1, 0.6, "This is for research purposes only.", fontsize=8)
        plt.text(0.1, 0.5, "Not a substitute for professional", fontsize=8)
        plt.text(0.1, 0.4, "medical diagnosis.", fontsize=8)
        plt.text(0.1, 0.3, "Always consult healthcare providers.", fontsize=8)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return True
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Handle file upload or URL input and analysis - returns JSON for AJAX"""
    # Check if this is a file upload or URL input
    if 'image_url' in request.form and request.form['image_url'].strip():
        # URL input mode
        image_url = request.form['image_url'].strip()
        print(f"Processing image from URL: {image_url}")

        # Basic URL validation
        if not image_url.startswith(('http://', 'https://')):
            return {'error': 'Invalid URL. Please provide a valid http or https URL.'}, 400

        # Check if URL points to an image file
        url_lower = image_url.lower()
        if not any(url_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']):
            return {'error': 'Invalid image URL. URL must point directly to an image file (PNG, JPG, JPEG, TIF, TIFF).'}, 400

        try:
            # Download image from URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                return {'error': 'URL does not point to a valid image file.'}, 400

            # Create temporary file for the downloaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp', dir=app.config['UPLOAD_FOLDER']) as tmp_file:
                tmp_file.write(response.content)
                filepath = tmp_file.name

            filename = 'downloaded_image'

        except requests.exceptions.RequestException as e:
            return {'error': f'Error downloading image from URL: {str(e)}'}, 400
        except Exception as e:
            return {'error': f'Error processing URL: {str(e)}'}, 500

    elif 'file' in request.files and request.files['file'].filename:
        # File upload mode
        file = request.files['file']

        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            return {'error': 'Invalid file type. Please upload a PNG, JPG, JPEG, TIF, or TIFF file.'}, 400
    else:
        return {'error': 'No file uploaded or URL provided'}, 400

    try:
        # Validate the image can be read by OpenCV
        image_test = cv2.imread(filepath)
        if image_test is None:
            # Clean up temporary file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            return {'error': 'Unable to read the image. Please ensure it is a valid image file.'}, 400

        # Process the image using full pipeline (like main.py)
        print(f"Processing image: {filename}")

        # Step 1: Preprocessing
        try:
            preprocess_results = preprocessor.preprocess_pipeline(
                filepath, save_steps=False, output_dir=None
            )
            preprocessed_image = preprocess_results['preprocessed']
        except:
            # Fallback if preprocessing fails
            image = cv2.imread(filepath)
            preprocessed_image = image

        # Step 2: Vessel Segmentation (proper method)
        try:
            # Save preprocessed image temporarily for segmentation
            temp_preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(temp_preprocessed_path, preprocessed_image)

            segmentation_results = segmenter.segment_vessels(
                temp_preprocessed_path, method='hybrid', save_results=False, output_dir=None
            )
            vessel_mask = segmentation_results['vessel_mask']

            # Clean up temp file
            if os.path.exists(temp_preprocessed_path):
                os.remove(temp_preprocessed_path)
        except:
            # Fallback to simple vessel detection if segmentation fails
            gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
            vessel_mask = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            vessel_mask = cv2.dilate(vessel_mask, kernel, iterations=1)

        # Step 3: Feature Extraction with proper vessel mask
        try:
            features = extractor.extract_features_from_array(
                preprocessed_image, vessel_mask_array=vessel_mask, save_results=False
            )
        except:
            # Fallback to basic feature extraction
            features = extractor.extract_features(filepath, save_results=False)

        # Make prediction if model is available
        if predictor and model_loaded:
            # Get combined predictions for both hypertension and heart disease
            combined_results = predictor.predict_combined_risks({
                'avr': features['avr'],
                'tortuosity': features['tortuosity'],
                'cdr': features['cdr'],
                'vessel_density': 0.85,  # Default value
                'optic_disc_ratio': 0.25  # Default value
            })

            # Return JSON response for AJAX
            response_data = {
                'avr': float(features['avr']),
                'tortuosity': float(features['tortuosity']),
                'cdr': float(features['cdr']),
                'hypertension': {
                    'prediction': combined_results['hypertension']['prediction'],
                    'probabilities': {
                        'Low Risk': float(combined_results['hypertension']['probabilities']['Low Risk']),
                        'Moderate Risk': float(combined_results['hypertension']['probabilities']['Moderate Risk']),
                        'High Risk': float(combined_results['hypertension']['probabilities']['High Risk'])
                    }
                },
                'heart_disease': {
                    'prediction': combined_results['heart_disease']['prediction'],
                    'probabilities': {
                        'Low Risk': float(combined_results['heart_disease']['probabilities']['Low Risk']),
                        'Moderate Risk': float(combined_results['heart_disease']['probabilities']['Moderate Risk']),
                        'High Risk': float(combined_results['heart_disease']['probabilities']['High Risk'])
                    }
                },
                'overall_risk': combined_results['overall_risk']['risk_level']
            }
        else:
            response_data = {
                'avr': float(features['avr']),
                'tortuosity': float(features['tortuosity']),
                'cdr': float(features['cdr']),
                'prediction': 'Model Not Available',
                'probabilities': {'Low Risk': 0.0, 'Moderate Risk': 0.0, 'High Risk': 0.0}
            }

        # Clean up file
        if os.path.exists(filepath):
            os.remove(filepath)

        return response_data

    except Exception as e:
        print(f"Error processing image: {e}")
        # Clean up file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return {'error': f'Error processing image: {str(e)}'}, 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis - legacy endpoint"""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Process the image
            print(f"Processing uploaded image: {filename}")

            # Extract features
            features = extractor.extract_features(filepath, save_results=False)

            # Make prediction if model is available
            if predictor and model_loaded:
                prediction = predictor.predict_risk({
                    'avr': features['avr'],
                    'tortuosity': features['tortuosity'],
                    'cdr': features['cdr']
                })
                features.update(prediction)
            else:
                features['hypertension_risk'] = 'Model Not Available'
                features['probabilities'] = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0}

            # Create visualization
            result_filename = f"result_{os.path.splitext(filename)[0]}.png"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

            # Ensure results directory exists
            os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

            visualization_created = create_visualization(filepath, features, result_path)

            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

            return render_template('result.html',
                                 filename=filename,
                                 features=features,
                                 result_image=result_filename if visualization_created else None,
                                 model_loaded=model_loaded)

        except Exception as e:
            print(f"Error processing image: {e}")
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('home'))

    else:
        flash('Invalid file type. Please upload a PNG, JPG, JPEG, TIF, or TIFF file.')
        return redirect(url_for('home'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    print("🚀 Starting Hypertension Detection Web Application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("📋 Upload retinal images to get instant hypertension risk analysis!")

    app.run(debug=True, host='0.0.0.0', port=5000)
