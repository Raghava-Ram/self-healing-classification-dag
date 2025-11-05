import torch
import logging
from typing import Dict, Any, Optional, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentClassifier:
    def __init__(self, model_path: str = "./fine_tuned_model", use_backup_model: bool = False):
        self.use_backup_model = use_backup_model
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Statistics tracking
        self.predictions: List[Dict] = []
        self.fallback_count = 0
        self.confidences: List[float] = []
        self.fallback_confidences: List[float] = []
        
        # Create stats directory
        self.stats_dir = Path("statistics")
        self.stats_dir.mkdir(exist_ok=True)
        
        # Load model and tokenizer
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Initialize backup model if needed
        if self.use_backup_model:
            try:
                logger.info("Initializing backup zero-shot model...")
                self.backup_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Successfully initialized backup zero-shot model")
            except Exception as e:
                logger.error(f"Error initializing backup model: {e}")
                self.use_backup_model = False
        
        logger.info(f"Initialized SentimentClassifier with model from {model_path}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction using the fine-tuned model."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Convert tensor to Python scalar
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            
            # Map predicted class to label (0 -> Negative, 1 -> Positive)
            label = "Positive" if predicted_class == 1 else "Negative"
            
            # Print the inference node output in the expected format
            print(f"[InferenceNode] Predicted label: {label} | Confidence: {int(confidence*100)}%")
            
            # Track statistics
            self.confidences.append(confidence)
            self.predictions.append({
                "text": text,
                "label": label,
                "confidence": confidence,
                "is_fallback": False,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "text": text,
                "label": label,
                "confidence": confidence,
                "is_fallback": False
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def fallback_prediction(self, text: str, original_confidence: float) -> Dict[str, Any]:
        """Handle fallback prediction with user input or backup model."""
        self.fallback_count += 1
        self.fallback_confidences.append(original_confidence)
        final_label = None
        confidence = original_confidence
        
        try:
            if hasattr(self, 'backup_model') and self.use_backup_model:
                logger.info("Using backup model for fallback")
                candidate_labels = ["positive", "negative"]
                result = self.backup_model(text, candidate_labels)
                final_label = result["labels"][0].capitalize()
                confidence = result["scores"][0]
                logger.info(f"Backup model predicted: {final_label} (confidence: {confidence:.4f})")
                print(f"[Fallback] Used backup model. Predicted: {final_label} (confidence: {confidence:.2f})")
            else:
                print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
                print("[FallbackNode] Could you clarify your intent? Was this a negative review?")
                user_input = input("User: ").strip().lower()
                
                if any(word in user_input for word in ["yes", "negative", "not good", "bad"]):
                    final_label = "Negative"
                else:
                    final_label = "Positive"
                    
                confidence = 1.0  # User input is considered 100% confident
                logger.info(f"User provided label: {final_label}")
                print(f"Final Label: {final_label} (Corrected via user clarification)")
            
            # Track statistics
            self.confidences.append(confidence)
            self.predictions.append({
                "text": text,
                "label": final_label,
                "confidence": confidence,
                "is_fallback": True,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "text": text,
                "label": final_label,
                "confidence": confidence,
                "is_fallback": True,
                "original_confidence": original_confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during fallback prediction: {e}")
            raise

    def generate_statistics(self):
        """Generate and save statistics and visualizations."""
        if not self.predictions:
            logger.warning("No predictions to generate statistics from")
            return
            
        try:
            # Create DataFrame from predictions
            df = pd.DataFrame(self.predictions)
            
            # 1. Confidence Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(self.confidences, bins=20, alpha=0.7, color='skyblue')
            plt.axvline(x=0.7, color='red', linestyle='--', label='Confidence Threshold')
            plt.title('Distribution of Prediction Confidences')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(self.stats_dir / 'confidence_distribution.png')
            plt.close()
            
            # 2. Fallback Statistics
            if self.fallback_count > 0:
                fallback_rate = (self.fallback_count / len(self.predictions)) * 100
                
                plt.figure(figsize=(10, 6))
                plt.pie(
                    [100 - fallback_rate, fallback_rate],
                    labels=['Direct Predictions', 'Fallbacks'],
                    autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral']
                )
                plt.title('Prediction Method Distribution')
                plt.savefig(self.stats_dir / 'fallback_distribution.png')
                plt.close()
                
                # Save statistics to file
                with open(self.stats_dir / 'statistics.txt', 'w') as f:
                    f.write(f"Total Predictions: {len(self.predictions)}\n")
                    f.write(f"Fallback Count: {self.fallback_count}\n")
                    f.write(f"Fallback Rate: {fallback_rate:.2f}%\n")
                    f.write(f"Average Confidence: {np.mean(self.confidences):.4f}\n")
                    
                    if self.fallback_confidences:
                        f.write(f"Average Fallback Confidence: {np.mean(self.fallback_confidences):.4f}\n")
            
            logger.info(f"Statistics saved to {self.stats_dir}")
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Healing Classification DAG")
    parser.add_argument("--use-backup-model", action="store_true", 
                       help="Use a backup model instead of user input for fallback")
    parser.add_argument("--model-path", type=str, default="./fine_tuned_model",
                       help="Path to the fine-tuned model")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                       help="Confidence threshold for fallback (default: 0.7)")
    parser.add_argument("--stats-interval", type=int, default=5,
                       help="Generate statistics after N predictions (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = SentimentClassifier(
            model_path=args.model_path,
            use_backup_model=args.use_backup_model
        )
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        return
    
    print("\n=== Self-Healing Classification DAG ===")
    print("Type 'exit' or press Ctrl+C to quit\n")
    
    try:
        while True:
            text = input("Enter your text for sentiment analysis: ")
            if text.lower() in ['exit', 'quit']:
                break
                
            if not text.strip():
                continue
            
            # Get initial prediction
            result = classifier.predict(text)
            
            # Check confidence
            if result["confidence"] < args.confidence_threshold:
                logger.info(f"Low confidence prediction ({result['confidence']:.4f}), triggering fallback...")
                result = classifier.fallback_prediction(text, result["confidence"])
            
            # For high confidence predictions, just show the result
            if result['confidence'] >= args.confidence_threshold:
                print(f"Final Label: {result['label']} (Confidence: {int(result['confidence']*100)}%)")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    print("\nLogs have been saved to 'logs.txt'")

# END is imported from langgraph

if __name__ == "__main__":
    main()
