"""
Visualization Script
Creates graphs and charts for TOPSIS results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load TOPSIS results"""
    output_file = 'results/topsis_output.csv'
    
    if not os.path.exists(output_file):
        print("Error: TOPSIS output not found!")
        print("Please run run_topsis.py first.")
        return None
    
    df = pd.read_csv(output_file)
    df = df.sort_values('Rank')
    return df

def plot_topsis_scores(df):
    """Bar chart of TOPSIS scores"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df['Model'], df['Topsis Score'], color=colors)
    
    # Add rank labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['Topsis Score'] + 0.01, i, 
                f"Rank {int(row['Rank'])}", 
                va='center', fontweight='bold')
    
    ax.set_xlabel('TOPSIS Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('TOPSIS Scores - Model Ranking\nText Sentence Similarity', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/topsis_scores_ranking.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/topsis_scores_ranking.png")
    plt.close()

def plot_criteria_comparison(df):
    """Radar chart comparing top 3 models"""
    # Normalize data for radar chart
    top3 = df.head(3).copy()
    
    criteria = ['Spearman_Correlation', 'Pearson_Correlation', 
                'Inference_Speed', 'Model_Size_MB', 'Memory_Usage_MB']
    
    # Normalize to 0-1 scale
    normalized = top3[criteria].copy()
    for col in criteria:
        min_val = df[col].min()
        max_val = df[col].max()
        # For size and memory (cost criteria), invert the normalization
        if col in ['Model_Size_MB', 'Memory_Usage_MB']:
            normalized[col] = 1 - (top3[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = (top3[col] - min_val) / (max_val - min_val)
    
    # Create radar chart
    labels = ['Spearman\nCorr', 'Pearson\nCorr', 'Inference\nSpeed', 
              'Model\nSize*', 'Memory\nUsage*']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (i, row) in enumerate(top3.iterrows()):
        values = normalized.iloc[idx].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=f"Rank {int(row['Rank'])}: {row['Model']}", 
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Models - Criteria Comparison\n(* inverted for cost criteria)',
                 fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/top3_radar_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/top3_radar_chart.png")
    plt.close()

def plot_criteria_heatmap(df):
    """Heatmap of all criteria"""
    criteria = ['Spearman_Correlation', 'Pearson_Correlation', 
                'Inference_Speed', 'Model_Size_MB', 'Memory_Usage_MB']
    
    # Prepare data
    heatmap_data = df[['Model'] + criteria].set_index('Model')
    
    # Normalize for better visualization
    normalized_data = heatmap_data.copy()
    for col in criteria:
        min_val = heatmap_data[col].min()
        max_val = heatmap_data[col].max()
        # Invert for cost criteria
        if col in ['Model_Size_MB', 'Memory_Usage_MB']:
            normalized_data[col] = 1 - (heatmap_data[col] - min_val) / (max_val - min_val)
        else:
            normalized_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
    
    # Rename columns for display
    normalized_data.columns = ['Spearman', 'Pearson', 'Speed', 
                                'Size (inv)', 'Memory (inv)']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(normalized_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Normalized Score'}, ax=ax, 
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Performance Heatmap - All Models\n(Normalized scores, higher is better)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Criteria', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/criteria_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/criteria_heatmap.png")
    plt.close()

def plot_performance_scatter(df):
    """Scatter plot: Accuracy vs Speed"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use average of correlations for accuracy
    df['Avg_Correlation'] = (df['Spearman_Correlation'] + df['Pearson_Correlation']) / 2
    
    # Create scatter plot
    scatter = ax.scatter(df['Inference_Speed'], df['Avg_Correlation'], 
                        s=1000/df['Model_Size_MB']*100, 
                        c=df['Topsis Score'], 
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for idx, row in df.iterrows():
        ax.annotate(f"{row['Model']}\n(Rank {int(row['Rank'])})", 
                   (row['Inference_Speed'], row['Avg_Correlation']),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                   fontsize=8, ha='left')
    
    ax.set_xlabel('Inference Speed (sentences/sec)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Correlation (Accuracy)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Accuracy vs Speed\n(Bubble size = 1/Model Size)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TOPSIS Score', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/accuracy_vs_speed.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/accuracy_vs_speed.png")
    plt.close()

def plot_criteria_weights():
    """Pie chart of TOPSIS criteria weights"""
    criteria = ['Spearman\nCorrelation', 'Pearson\nCorrelation', 
                'Inference\nSpeed', 'Model\nSize', 'Memory\nUsage']
    weights = [30, 25, 20, 15, 10]
    colors = sns.color_palette('Set3', len(criteria))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(weights, labels=criteria, autopct='%1.0f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('TOPSIS Criteria Weights Distribution',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/criteria_weights.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/criteria_weights.png")
    plt.close()

def create_comparison_table(df):
    """Create a detailed comparison table"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Rank', 'Model', 'TOPSIS\nScore', 'Spearman', 
                      'Pearson', 'Speed\n(sent/s)', 'Size\n(M)', 'Memory\n(MB)'])
    
    for idx, row in df.iterrows():
        table_data.append([
            int(row['Rank']),
            row['Model'],
            f"{row['Topsis Score']:.4f}",
            f"{row['Spearman_Correlation']:.4f}",
            f"{row['Pearson_Correlation']:.4f}",
            f"{row['Inference_Speed']:.1f}",
            f"{row['Model_Size_MB']:.1f}",
            f"{row['Memory_Usage_MB']:.1f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.08, 0.28, 0.12, 0.10, 0.10, 0.12, 0.10, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(8):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best model row
    for i in range(8):
        table[(1, i)].set_facecolor('#FFD700')
        table[(1, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(2, len(table_data)):
        for j in range(8):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Comparison Table - TOPSIS Results\nText Sentence Similarity',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/comparison_table.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/comparison_table.png")
    plt.close()

def main():
    print("="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\nGenerating graphs for {len(df)} models...")
    print()
    
    # Create all visualizations
    plot_topsis_scores(df)
    plot_criteria_comparison(df)
    plot_criteria_heatmap(df)
    plot_performance_scatter(df)
    plot_criteria_weights()
    create_comparison_table(df)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS CREATED!")
    print("="*60)
    print("\nGenerated files in results/:")
    print("  1. topsis_scores_ranking.png - Bar chart of rankings")
    print("  2. top3_radar_chart.png - Radar comparison of top 3")
    print("  3. criteria_heatmap.png - Performance heatmap")
    print("  4. accuracy_vs_speed.png - Scatter plot")
    print("  5. criteria_weights.png - Pie chart of weights")
    print("  6. comparison_table.png - Detailed comparison table")

if __name__ == "__main__":
    main()
