"""
TOPSIS Analysis Script
Uses the custom Topsis-Prabhsimar-102483078 package to rank models
"""

import pandas as pd
import numpy as np
import subprocess
import os
import sys

def prepare_topsis_input():
    """
    Prepare the input CSV file for TOPSIS analysis
    """
    print("="*60)
    print("PREPARING DATA FOR TOPSIS ANALYSIS")
    print("="*60)
    
    # Load evaluation results
    results_file = 'results/model_evaluation_results.csv'
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        print("Please run evaluate_models.py first.")
        sys.exit(1)
    
    df = pd.read_csv(results_file)
    
    print(f"\nLoaded {len(df)} models from evaluation results")
    print("\nOriginal Data:")
    print(df.to_string(index=False))
    
    # Prepare TOPSIS input format
    # Columns: Model, Spearman, Pearson, Speed, Size, Memory
    topsis_input = pd.DataFrame({
        'Model': df['model_name'],
        'Spearman_Correlation': df['spearman_correlation'],
        'Pearson_Correlation': df['pearson_correlation'],
        'Inference_Speed': df['inference_speed'],
        'Model_Size_MB': df['model_size'],
        'Memory_Usage_MB': df['memory_usage']
    })
    
    # Save to CSV
    input_file = 'results/topsis_input.csv'
    topsis_input.to_csv(input_file, index=False)
    
    print(f"\n✓ TOPSIS input prepared: {input_file}")
    print("\nTOPSIS Input Data:")
    print(topsis_input.to_string(index=False))
    
    return input_file, topsis_input

def run_topsis_analysis(input_file):
    """
    Run TOPSIS using the custom package
    """
    print("\n" + "="*60)
    print("RUNNING TOPSIS ANALYSIS")
    print("="*60)
    
    # Define TOPSIS parameters based on our criteria
    weights = "0.30,0.25,0.20,0.15,0.10"
    impacts = "+,+,+,-,-"  # + for benefit (maximize), - for cost (minimize)
    output_file = 'results/topsis_output.csv'
    
    print("\nTOPSIS Configuration:")
    print(f"  Weights: {weights}")
    print(f"    - Spearman Correlation: 30% (maximize)")
    print(f"    - Pearson Correlation: 25% (maximize)")
    print(f"    - Inference Speed: 20% (maximize)")
    print(f"    - Model Size: 15% (minimize)")
    print(f"    - Memory Usage: 10% (minimize)")
    print(f"\n  Impacts: {impacts}")
    print(f"    + = Benefit (higher is better)")
    print(f"    - = Cost (lower is better)")
    
    # Run TOPSIS command
    print("\nExecuting TOPSIS command...")
    cmd = f'topsis "{input_file}" "{weights}" "{impacts}" "{output_file}"'
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✓ TOPSIS analysis completed successfully!")
        
        if result.stdout:
            print("\nTOPSIS Output:")
            print(result.stdout)
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error running TOPSIS: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def display_results(output_file):
    """
    Display and analyze TOPSIS results
    """
    print("\n" + "="*60)
    print("TOPSIS RESULTS")
    print("="*60)
    
    df = pd.read_csv(output_file)
    
    # Sort by rank
    df_sorted = df.sort_values('Rank')
    
    print("\n" + "="*60)
    print("FINAL MODEL RANKING")
    print("="*60)
    print()
    print(df_sorted.to_string(index=False))
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Get best model
    best_model = df_sorted.iloc[0]
    print(f"\n🏆 BEST MODEL (Rank 1):")
    print(f"   Model: {best_model['Model']}")
    print(f"   TOPSIS Score: {best_model['Topsis Score']:.6f}")
    print(f"   Spearman Correlation: {best_model['Spearman_Correlation']:.4f}")
    print(f"   Pearson Correlation: {best_model['Pearson_Correlation']:.4f}")
    print(f"   Inference Speed: {best_model['Inference_Speed']:.2f} sentences/sec")
    print(f"   Model Size: {best_model['Model_Size_MB']:.1f} M parameters")
    print(f"   Memory Usage: {best_model['Memory_Usage_MB']:.2f} MB")
    
    # Get worst model
    worst_model = df_sorted.iloc[-1]
    print(f"\n📊 LOWEST RANKED MODEL (Rank {len(df)}):")
    print(f"   Model: {worst_model['Model']}")
    print(f"   TOPSIS Score: {worst_model['Topsis Score']:.6f}")
    
    # Top 3 models
    print(f"\n🥇 TOP 3 MODELS:")
    for i, row in df_sorted.head(3).iterrows():
        print(f"\n   Rank {int(row['Rank'])}: {row['Model']}")
        print(f"   TOPSIS Score: {row['Topsis Score']:.6f}")
    
    return df_sorted

def save_summary(df_sorted):
    """
    Save analysis summary
    """
    summary_file = 'results/topsis_analysis_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEXT SENTENCE SIMILARITY - TOPSIS ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("TOPSIS CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write("Criteria and Weights:\n")
        f.write("  1. Spearman Correlation (30%) - Maximize\n")
        f.write("  2. Pearson Correlation (25%) - Maximize\n")
        f.write("  3. Inference Speed (20%) - Maximize\n")
        f.write("  4. Model Size (15%) - Minimize\n")
        f.write("  5. Memory Usage (10%) - Minimize\n\n")
        
        f.write("FINAL RANKING\n")
        f.write("-" * 70 + "\n\n")
        
        for idx, row in df_sorted.iterrows():
            f.write(f"Rank {int(row['Rank'])}: {row['Model']}\n")
            f.write(f"  TOPSIS Score: {row['Topsis Score']:.6f}\n")
            f.write(f"  Spearman: {row['Spearman_Correlation']:.4f}\n")
            f.write(f"  Pearson: {row['Pearson_Correlation']:.4f}\n")
            f.write(f"  Speed: {row['Inference_Speed']:.2f} sent/sec\n")
            f.write(f"  Size: {row['Model_Size_MB']:.1f}M params\n")
            f.write(f"  Memory: {row['Memory_Usage_MB']:.2f} MB\n\n")
        
        best = df_sorted.iloc[0]
        f.write("="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n")
        f.write(f"Based on TOPSIS multi-criteria analysis, the best model is:\n")
        f.write(f"  {best['Model']}\n\n")
        f.write(f"This model achieves the highest TOPSIS score ({best['Topsis Score']:.6f})\n")
        f.write(f"by balancing all evaluation criteria effectively.\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    return summary_file

def main():
    print("="*60)
    print("TOPSIS ANALYSIS FOR SENTENCE SIMILARITY MODELS")
    print("Using: Topsis-Prabhsimar-102483078 package")
    print("="*60)
    
    # Step 1: Prepare input
    input_file, topsis_input = prepare_topsis_input()
    
    # Step 2: Run TOPSIS
    output_file = run_topsis_analysis(input_file)
    
    # Step 3: Display results
    df_sorted = display_results(output_file)
    
    # Step 4: Save summary
    summary_file = save_summary(df_sorted)
    
    print("\n" + "="*60)
    print("TOPSIS ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. {input_file}")
    print(f"  2. {output_file}")
    print(f"  3. {summary_file}")
    print("\nNext step: Create visualizations and GitHub documentation")

if __name__ == "__main__":
    main()
