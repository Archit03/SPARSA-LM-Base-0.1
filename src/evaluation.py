import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import numpy as np
from tqdm import tqdm
from src.model import Transformer
from src.tokenizer import MedicalTokenizer
from src.utils import setup_logging
import logging
import yaml
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.training import CombinedDataset

class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        tokenizer: MedicalTokenizer,
        device: torch.device,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_predictions(self) -> Tuple[List[int], List[int], List[float]]:
        """Get model predictions and probabilities."""
        all_predictions = []
        all_labels = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(self.dataloader, desc="Evaluating")
            for batch in progress_bar:
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    # Get predictions and probabilities
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(outputs.logits, dim=-1)

                    # Collect predictions, labels, and probabilities
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                    all_probabilities.extend(probabilities.cpu().numpy().max(axis=-1).flatten())

                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue

        return all_predictions, all_labels, all_probabilities

    def plot_confusion_matrix(self, labels: List[int], predictions: List[int]) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

    def plot_score_distribution(self, probabilities: List[float], labels: List[int]) -> None:
        """Plot and save prediction probability distribution."""
        plt.figure(figsize=(10, 6))
        for label in set(labels):
            mask = np.array(labels) == label
            plt.hist(np.array(probabilities)[mask], bins=50, alpha=0.5, 
                    label=f'Class {label}')
        plt.title('Prediction Probability Distribution by Class')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(self.output_dir / 'score_distribution.png')
        plt.close()

    def analyze_errors(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: List[float],
        n_samples: int = 10
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = {
            "high_confidence_errors": [],
            "low_confidence_correct": []
        }

        for pred, label, prob in zip(predictions, labels, probabilities):
            if pred != label and prob > 0.9:
                errors["high_confidence_errors"].append({
                    "predicted": pred,
                    "actual": label,
                    "confidence": prob
                })
            elif pred == label and prob < 0.6:
                errors["low_confidence_correct"].append({
                    "predicted": pred,
                    "actual": label,
                    "confidence": prob
                })

        # Sort and limit samples
        errors["high_confidence_errors"] = sorted(
            errors["high_confidence_errors"],
            key=lambda x: x["confidence"],
            reverse=True
        )[:n_samples]

        errors["low_confidence_correct"] = sorted(
            errors["low_confidence_correct"],
            key=lambda x: x["confidence"]
        )[:n_samples]

        return errors

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model and compute comprehensive metrics."""
        try:
            predictions, labels, probabilities = self._get_predictions()

            # Basic metrics
            metrics = {
                "accuracy": accuracy_score(labels, predictions),
                "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
                "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
                "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
            }

            # Calculate per-class metrics
            classification_rep = classification_report(labels, predictions, output_dict=True)
            metrics["per_class_metrics"] = classification_rep

            # Calculate ROC AUC if binary classification
            unique_labels = set(labels)
            if len(unique_labels) == 2:
                metrics["roc_auc"] = roc_auc_score(labels, probabilities)

            # Generate visualizations
            self.plot_confusion_matrix(labels, predictions)
            self.plot_score_distribution(probabilities, labels)

            # Analyze errors
            error_analysis = self.analyze_errors(predictions, labels, probabilities)
            metrics["error_analysis"] = error_analysis

            # Calculate confidence statistics
            metrics["confidence_stats"] = {
                "mean_confidence": np.mean(probabilities),
                "median_confidence": np.median(probabilities),
                "std_confidence": np.std(probabilities),
            }

            # Log results
            logging.info(f"Evaluation metrics: {metrics}")
            
            # Save metrics to file
            metrics_path = self.output_dir / 'metrics.yaml'
            with open(metrics_path, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)

            return metrics

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

def evaluate_model(config_path: str, model_path: str, tokenizer_path: str, output_dir: str):
    """Main evaluation function with enhanced error handling and logging."""
    try:
        # Set up logging
        setup_logging()
        
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Initialize tokenizer
        tokenizer = MedicalTokenizer(tokenizer_path)
        
        # Load model
        model = Transformer(
            src_vocab_size=tokenizer.tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.tokenizer.vocab_size,
            **config.get("model_params", {})
        )
        
        # Load model weights with error handling
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model weights: {str(e)}")
            raise
        
        model.eval()
        
        # Device handling with logging
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model.to(device)
        
        # Load evaluation dataset
        eval_dataset = CombinedDataset(
            config["eval_dataset_config"],
            tokenizer,
            max_length=config.get("max_length", 512)
        )
        
        # Create dataloader with error handling
        try:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
            )
        except Exception as e:
            logging.error(f"Error creating DataLoader: {str(e)}")
            raise
        
        # Initialize evaluator
        evaluator = Evaluator(
            model=model,
            dataloader=eval_loader,
            tokenizer=tokenizer,
            device=device,
            output_dir=output_dir
        )
        
        # Perform evaluation
        metrics = evaluator.evaluate()
        logging.info("Evaluation completed successfully.")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise