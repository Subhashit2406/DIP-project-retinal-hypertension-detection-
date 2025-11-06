#!/usr/bin/env python3
"""
Hypertension & Heart Disease Detection from Retinal Fundus Images
Main pipeline script that integrates all components
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Import our custom modules
from src.preprocessing import RetinalImagePreprocessor
from src.vessel_segmentation import VesselSegmenter
from src.feature_extraction import FeatureExtractor
from src.ml_model import HypertensionRiskPredictor

class HypertensionDetectionPipeline:
    """Complete pipeline for hypertension detection from retinal images"""

    def __init__(self):
        """Initialize all components"""
        self.preprocessor = RetinalImagePreprocessor()
        self.segmenter = VesselSegmenter()
        self.extractor = FeatureExtractor()
        self.predictor = None

        # Load trained model if available
        self.load_trained_model()

    def load_trained_model(self):
        """Load pre-trained ML model"""
        model_path = "models/hypertension_risk_predictor.pkl"
        if os.path.exists(model_path):
            try:
                self.predictor = HypertensionRiskPredictor()
                self.predictor.load_model(model_path)
                print("✓ Pre-trained model loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load pre-trained model: {e}")
                print("Training new model...")
                self.train_model()
        else:
            print("No pre-trained model found. Training new model...")
            self.train_model()

    def train_model(self):
        """Train ML model"""
        if self.predictor is None:
            self.predictor = HypertensionRiskPredictor()

        # Generate synthetic dataset and train
        dataset = self.predictor.generate_synthetic_dataset(n_samples=2000)

        # Prepare data
        X, y = self.predictor.prepare_data(dataset)

        # Train models (this will select the best one)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        trained_models = self.predictor.train_models(X_train, y_train)

        # Save the best model
        os.makedirs("models", exist_ok=True)
        self.predictor.save_model("models/hypertension_risk_predictor.pkl")

        print("✓ Model trained and saved successfully")

    def process_single_image(self, image_path, output_dir="results/pipeline", save_steps=True):
        """
        Process a single retinal image through the complete pipeline

        Args:
            image_path: Path to input retinal image
            output_dir: Directory to save results
            save_steps: Whether to save intermediate steps

        Returns:
            Dictionary with all results
        """
        print(f"\n🔍 Processing image: {image_path}")
        print("=" * 50)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'image_path': image_path,
            'processing_steps': {}
        }

        try:
            # Step 1: Preprocessing
            print("📷 Step 1: Image Preprocessing...")
            preprocess_results = self.preprocessor.preprocess_pipeline(
                image_path,
                save_steps=save_steps,
                output_dir=str(output_path / "preprocessing")
            )
            results['processing_steps']['preprocessing'] = preprocess_results

            # Step 2: Vessel Segmentation
            print("🩸 Step 2: Blood Vessel Segmentation...")
            preprocessed_image = preprocess_results['preprocessed']

            # Save preprocessed image temporarily
            temp_preprocessed_path = str(output_path / "temp_preprocessed.png")
            cv2.imwrite(temp_preprocessed_path, preprocessed_image)

            segmentation_results = self.segmenter.segment_vessels(
                temp_preprocessed_path,
                method='hybrid',
                save_results=save_steps,
                output_dir=str(output_path / "segmentation")
            )
            results['processing_steps']['segmentation'] = segmentation_results

            # Clean up temp file
            if os.path.exists(temp_preprocessed_path):
                os.remove(temp_preprocessed_path)

            # Step 3: Feature Extraction
            print("📊 Step 3: Medical Feature Extraction...")
            feature_results = self.extractor.extract_features(
                image_path,
                save_results=save_steps,
                output_dir=str(output_path / "features")
            )
            results['processing_steps']['features'] = feature_results

            # Step 4: Risk Prediction
            print("🤖 Step 4: Hypertension Risk Prediction...")
            if self.predictor and self.predictor.best_model:
                prediction_results = self.predictor.predict_risk({
                    'avr': feature_results['avr'],
                    'tortuosity': feature_results['tortuosity'],
                    'cdr': feature_results['cdr']
                })
                results['processing_steps']['prediction'] = prediction_results
            else:
                results['processing_steps']['prediction'] = {
                    'error': 'Model not available'
                }

            # Step 5: Generate Report
            print("📋 Step 5: Generating Final Report...")
            report = self.generate_report(results)
            results['report'] = report

            # Save report with UTF-8 encoding
            with open(str(output_path / "analysis_report.txt"), 'w', encoding='utf-8') as f:
                f.write(report)

            print("✅ Processing completed successfully!")
            return results

        except Exception as e:
            print(f"❌ Error processing image: {e}")
            results['error'] = str(e)
            return results

    def generate_report(self, results):
        """Generate a comprehensive analysis report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("HYPERTENSION & HEART DISEASE DETECTION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Patient Information
        image_path = results['image_path']
        report_lines.append(f"📁 Image Analyzed: {Path(image_path).name}")
        report_lines.append("")

        # Extracted Features
        if 'features' in results['processing_steps']:
            features = results['processing_steps']['features']
            report_lines.append("📊 EXTRACTED MEDICAL FEATURES:")
            report_lines.append("-" * 30)
            report_lines.append(f"AVR: {features['avr']:.3f}")
            report_lines.append(f"Tortuosity: {features['tortuosity']:.3f}")
            report_lines.append(f"CDR: {features['cdr']:.3f}")
            report_lines.append("")

        # Risk Assessment
        if 'prediction' in results['processing_steps']:
            prediction = results['processing_steps']['prediction']
            if 'prediction' in prediction:
                report_lines.append("🎯 HYPERTENSION RISK ASSESSMENT:")
                report_lines.append("-" * 30)
                report_lines.append(f"Risk Level: {prediction['prediction']}")
                report_lines.append("")
                report_lines.append("Probability Distribution:")
                for risk_level, prob in prediction['probabilities'].items():
                    report_lines.append(f"{risk_level}: {prob * 100:.1f}%")
                report_lines.append("")

        # Medical Interpretation
        report_lines.append("🏥 MEDICAL INTERPRETATION:")
        report_lines.append("-" * 30)

        if 'features' in results['processing_steps']:
            features = results['processing_steps']['features']
            avr = features['avr']
            tortuosity = features['tortuosity']
            cdr = features['cdr']

            # AVR Interpretation
            if avr < 0.66:
                report_lines.append("• ARTERIOVENOUS RATIO: LOW (< 0.66) - Indicates potential hypertension")
            elif avr < 0.8:
                report_lines.append("• ARTERIOVENOUS RATIO: BORDERLINE - Monitor blood pressure")
            else:
                report_lines.append("• ARTERIOVENOUS RATIO: NORMAL - No immediate concern")

            # Tortuosity Interpretation
            if tortuosity > 1.2:
                report_lines.append("• VESSEL TORTUOSITY: HIGH (> 1.2) - Indicates cardiovascular stress")
            elif tortuosity > 1.1:
                report_lines.append("• VESSEL TORTUOSITY: ELEVATED - Monitor cardiovascular health")
            else:
                report_lines.append("• VESSEL TORTUOSITY: NORMAL - Good vascular health")

            # CDR Interpretation
            if cdr > 0.6:
                report_lines.append("• CUP-TO-DISC RATIO: HIGH (> 0.6) - Potential glaucoma risk")
            elif cdr < 0.3:
                report_lines.append("• CUP-TO-DISC RATIO: LOW (< 0.3) - Monitor for changes")
            else:
                report_lines.append("• CUP-TO-DISC RATIO: NORMAL - No immediate concern")

        report_lines.append("")
        report_lines.append("⚠️  IMPORTANT NOTES:")
        report_lines.append("-" * 30)
        report_lines.append("• This analysis is for research purposes only")
        report_lines.append("• Consult with a qualified medical professional for diagnosis")
        report_lines.append("• Regular medical check-ups are essential for cardiovascular health")
        report_lines.append("• This tool complements, but does not replace, clinical examination")

        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("Report generated by Hypertension Detection System")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def create_summary_visualization(self, results, output_path):
        """Create a summary visualization of all results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # Original image
        if 'preprocessing' in results['processing_steps']:
            original = results['processing_steps']['preprocessing']['original']
            if len(original.shape) == 3:
                axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            else:
                axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

        # Preprocessed image
        if 'preprocessing' in results['processing_steps']:
            preprocessed = results['processing_steps']['preprocessing']['preprocessed']
            axes[1].imshow(preprocessed, cmap='gray')
            axes[1].set_title('Preprocessed')
            axes[1].axis('off')

        # Vessel segmentation
        if 'segmentation' in results['processing_steps']:
            overlay = results['processing_steps']['segmentation']['overlay']
            axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Vessel Segmentation')
            axes[2].axis('off')

        # Feature visualization
        if 'features' in results['processing_steps']:
            features = results['processing_steps']['features']
            axes[3].bar(['AVR', 'Tortuosity', 'CDR'],
                       [features['avr'], features['tortuosity'], features['cdr']])
            axes[3].set_title('Extracted Features')
            axes[3].set_ylabel('Value')

        # Risk prediction
        if 'prediction' in results['processing_steps']:
            prediction = results['processing_steps']['prediction']
            if 'probabilities' in prediction:
                risk_levels = list(prediction['probabilities'].keys())
                probabilities = list(prediction['probabilities'].values())

                axes[4].bar(risk_levels, probabilities)
                axes[4].set_title('Risk Probabilities')
                axes[4].set_ylabel('Probability')
                axes[4].tick_params(axis='x', rotation=45)

        # Summary text
        axes[5].text(0.1, 0.8, "ANALYSIS SUMMARY", fontsize=14, fontweight='bold')
        axes[5].text(0.1, 0.6, f"Image: {Path(results['image_path']).name}", fontsize=10)

        if 'prediction' in results['processing_steps']:
            pred = results['processing_steps']['prediction']
            if 'prediction' in pred:
                axes[5].text(0.1, 0.4, f"Risk: {pred['prediction']}", fontsize=12, color='red')

        axes[5].set_xlim(0, 1)
        axes[5].set_ylim(0, 1)
        axes[5].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_batch_summary(self, batch_results, output_dir):
        """Generate summary for batch processing"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("BATCH PROCESSING SUMMARY - HYPERTENSION DETECTION")
        summary_lines.append("=" * 80)
        summary_lines.append("")

        total_images = len(batch_results)
        successful = sum(1 for r in batch_results if 'error' not in r)
        failed = total_images - successful

        summary_lines.append(f"Total Images Processed: {total_images}")
        summary_lines.append(f"Successful: {successful}")
        summary_lines.append(f"Failed: {failed}")
        summary_lines.append("")

        # Risk distribution
        risk_counts = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0}
        for result in batch_results:
            if 'processing_steps' in result and 'prediction' in result['processing_steps']:
                pred = result['processing_steps']['prediction']
                if 'prediction' in pred:
                    risk_counts[pred['prediction']] += 1

        summary_lines.append("RISK DISTRIBUTION:")
        summary_lines.append("-" * 20)
        for risk_level, count in risk_counts.items():
            percentage = (count / successful) * 100 if successful > 0 else 0
            summary_lines.append(f"{risk_level}: {count} ({percentage:.1f}%)")
        summary_lines.append("")

        # Individual results
        summary_lines.append("INDIVIDUAL RESULTS:")
        summary_lines.append("-" * 20)
        for i, result in enumerate(batch_results, 1):
            image_name = Path(result['image_path']).name
            if 'error' in result:
                summary_lines.append(f"{i:2d}. {image_name} - ERROR: {result['error']}")
            else:
                pred = result['processing_steps']['prediction']
                risk = pred.get('prediction', 'Unknown') if 'prediction' in pred else 'Unknown'
                summary_lines.append(f"{i:2d}. {image_name} - Risk: {risk}")

        summary_lines.append("")
        summary_lines.append("=" * 80)

        return "\n".join(summary_lines)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Hypertension Detection from Retinal Images')
    parser.add_argument('--image', '-i', type=str, help='Path to retinal image to analyze')
    parser.add_argument('--batch', '-b', type=str, help='Path to directory containing multiple images')
    parser.add_argument('--output', '-o', type=str, default='results/pipeline',
                       help='Output directory for results')
    parser.add_argument('--train', action='store_true', help='Train new ML model')

    args = parser.parse_args()

    # Initialize pipeline
    print("🚀 Starting Hypertension Detection Pipeline...")
    pipeline = HypertensionDetectionPipeline()

    if args.train:
        print("🔧 Training new ML model...")
        pipeline.train_model()
        return

    if args.image:
        # Process single image
        results = pipeline.process_single_image(args.image, args.output)

        if 'report' in results:
            print("\n" + "="*60)
            print(results['report'])
            print("="*60)

        # Create summary visualization
        pipeline.create_summary_visualization(results, f"{args.output}/summary.png")
        print(f"\n📊 Summary visualization saved to {args.output}/summary.png")

    elif args.batch:
        # Process batch of images
        image_dir = Path(args.batch)
        if not image_dir.exists():
            print(f"❌ Directory not found: {args.batch}")
            return

        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = [f for f in image_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"❌ No image files found in {args.batch}")
            return

        print(f"📂 Processing {len(image_files)} images...")

        batch_results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔄 Processing image {i}/{len(image_files)}: {image_file.name}")
            results = pipeline.process_single_image(
                str(image_file),
                f"{args.output}/{image_file.stem}"
            )
            batch_results.append(results)

        # Create batch summary
        print("\n📋 Generating batch summary...")
        batch_summary = pipeline.generate_batch_summary(batch_results, args.output)
        with open(f"{args.output}/batch_summary.txt", 'w') as f:
            f.write(batch_summary)

        print(f"✅ Batch processing completed! Results saved to {args.output}")

    else:
        print("❌ Please specify either --image or --batch argument")
        print("Example: python main.py --image path/to/image.png")
        print("Example: python main.py --batch path/to/image/directory")

if __name__ == "__main__":
    main()
