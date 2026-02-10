"""
Model Evaluation Script
Evaluates all models on STS Benchmark dataset
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import time
import psutil
import os
from tqdm import tqdm
import json

from model_config import MODELS, get_model_names

class ModelEvaluator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.results = []
        
    def load_sts_benchmark(self):
        """Load STS Benchmark dataset"""
        print("\nLoading STS Benchmark dataset...")
        try:
            # Load the dataset
            dataset = load_dataset('mteb/stsbenchmark-sts')
            test_data = dataset['test']
            
            # Extract sentence pairs and scores
            sentence1 = test_data['sentence1']
            sentence2 = test_data['sentence2']
            scores = np.array(test_data['score']) / 5.0  # Normalize to 0-1
            
            print(f"Dataset loaded: {len(sentence1)} sentence pairs")
            return sentence1, sentence2, scores
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample dataset for demonstration...")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self):
        """Create a sample dataset if main dataset fails"""
        sentence1 = [
            "A man is eating food.",
            "A plane is taking off.",
            "A woman is playing violin.",
            "The cat is sleeping.",
            "A dog is running in the park."
        ] * 10
        
        sentence2 = [
            "A man is eating pasta.",
            "An airplane is taking off.",
            "A woman is playing music.",
            "The cat is resting.",
            "A dog is playing outside."
        ] * 10
        
        scores = np.array([0.8, 0.9, 0.7, 0.85, 0.75] * 10)
        
        return sentence1, sentence2, scores
    
    def evaluate_model(self, model_info, sentence1, sentence2, true_scores):
        """Evaluate a single model"""
        model_name = model_info['name']
        model_id = model_info['model_id']
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Model ID: {model_id}")
        print(f"{'='*60}")
        
        try:
            # Load model
            print("Loading model...")
            model = SentenceTransformer(model_id, device=self.device)
            
            # Measure memory usage
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Encode sentences and measure speed
            print("Encoding sentences...")
            start_time = time.time()
            
            embeddings1 = model.encode(sentence1, show_progress_bar=True, batch_size=32)
            embeddings2 = model.encode(sentence2, show_progress_bar=True, batch_size=32)
            
            end_time = time.time()
            
            # Calculate metrics
            # 1. Cosine similarity
            similarities = []
            for i in range(len(embeddings1)):
                sim = cosine_similarity(
                    embeddings1[i].reshape(1, -1),
                    embeddings2[i].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            similarities = np.array(similarities)
            
            # 2. Correlation metrics
            pearson_corr, _ = pearsonr(similarities, true_scores)
            spearman_corr, _ = spearmanr(similarities, true_scores)
            
            # 3. Speed (sentences per second)
            total_sentences = len(sentence1) * 2
            total_time = end_time - start_time
            speed = total_sentences / total_time
            
            # 4. Memory usage
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # 5. Model size (from config)
            model_size = model_info['parameters_millions']
            
            results = {
                'model_name': model_name,
                'model_id': model_id,
                'spearman_correlation': spearman_corr,
                'pearson_correlation': pearson_corr,
                'inference_speed': speed,
                'model_size': model_size,
                'memory_usage': memory_usage
            }
            
            print(f"\nResults:")
            print(f"  Spearman Correlation: {spearman_corr:.4f}")
            print(f"  Pearson Correlation: {pearson_corr:.4f}")
            print(f"  Inference Speed: {speed:.2f} sentences/sec")
            print(f"  Model Size: {model_size} M parameters")
            print(f"  Memory Usage: {memory_usage:.2f} MB")
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None
    
    def evaluate_all_models(self):
        """Evaluate all models in the configuration"""
        # Load dataset
        sentence1, sentence2, true_scores = self.load_sts_benchmark()
        
        # Evaluate each model
        for model_info in MODELS:
            result = self.evaluate_model(model_info, sentence1, sentence2, true_scores)
            if result:
                self.results.append(result)
        
        # Create results DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Save results
        print(f"\n{'='*60}")
        print("Saving results...")
        df_results.to_csv('results/model_evaluation_results.csv', index=False)
        
        # Save as JSON too
        with open('results/model_evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("Results saved to:")
        print("  - results/model_evaluation_results.csv")
        print("  - results/model_evaluation_results.json")
        
        # Display results table
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}\n")
        print(df_results.to_string(index=False))
        
        return df_results

def main():
    print("="*60)
    print("TEXT SENTENCE SIMILARITY - MODEL EVALUATION")
    print("="*60)
    
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_all_models()
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"\nTotal models evaluated: {len(results_df)}")
    print("\nNext step: Apply TOPSIS analysis to rank models")

if __name__ == "__main__":
    main()
