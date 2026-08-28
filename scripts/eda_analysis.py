"""
═══════════════════════════════════════════════════════════════════════════════
EXPLORATORY DATA ANALYSIS (EDA) - CLASSIFIED DATASET
═══════════════════════════════════════════════════════════════════════════════

PROJECT: Amharic-Oromo Translation NLP
PHASE: 1 - Data Preparation
INPUT FILE: data/final.csv
OUTPUT DIR: data/processed/

FOCUS: 
  ✓ Analyze text length and word distributions
  ✓ Analyze domain distribution and length per domain
  ✓ Evaluate source-to-target character/word length ratios
  ✓ Generate 3 comprehensive visualization charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# SETUP & PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "final.csv"

OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# DATA LOADING & STATS
# ============================================================================

def load_data():
    """Loads input file: data/final.csv"""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Required input dataset not found at: {INPUT_FILE}")
    
    print(f"📂 Loading classified dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    
    # Standardize column names
    col_map = {col: col.capitalize() for col in df.columns}
    df = df.rename(columns=col_map)
    
    # Required columns check
    for req in ['Amharic', 'Oromo']:
        if req not in df.columns:
            raise KeyError(f"Expected column '{req}' not found in {INPUT_FILE.name}")
            
    if 'Domain' not in df.columns:
        df['Domain'] = 'General'
        
    return df

def calculate_metrics(df):

    """Calculates text length and word counts statistics"""
    df['am_chars'] = df['Amharic'].fillna("").astype(str).str.len()
    df['or_chars'] = df['Oromo'].fillna("").astype(str).str.len()
    df['am_words'] = df['Amharic'].fillna("").astype(str).apply(lambda x: len(x.split()))
    df['or_words'] = df['Oromo'].fillna("").astype(str).apply(lambda x: len(x.split()))
    
    # Length ratio (Oromo / Amharic) to measure expansion
    df['char_ratio'] = df['or_chars'] / df['am_chars'].replace(0, np.nan)
    df['word_ratio'] = df['or_words'] / df['am_words'].replace(0, np.nan)
    
    return df

# ============================================================================
# SUMMARY REPORTING
# ============================================================================

def print_eda_summary(df):
    """Prints a structured text summary in terminal"""
    print("\n" + "="*80)
    print("AMHARIC-OROMO CLASSIFIED DATASET - EDA SUMMARY")
    print("="*80)
    print(f"Total Parallel Rows Analyzed: {len(df):,}\n")
    
    print("📌 TEXT LENGTH METRICS")
    print(f"  Amharic  - Avg Chars: {df['am_chars'].mean():.1f} | Avg Words: {df['am_words'].mean():.1f}")
    print(f"  Char Range: {df['am_chars'].min()} - {df['am_chars'].max()} | Word Range: {df['am_words'].min()} - {df['am_words'].max()}")
    print(f"  Afaan Oromo - Avg Chars: {df['or_chars'].mean():.1f} | Avg Words: {df['or_words'].mean():.1f}")
    print(f"Char Range: {df['or_chars'].min()} - {df['or_chars'].max()} | Word Range: {df['or_words'].min()} - {df['or_words'].max()}")
    
    print("\n📌 LENGTH EXPANSION RATIOS (Target / Source)")
    print(f"  Avg Character Ratio (Oromo/Amharic): {df['char_ratio'].mean():.2f}")
    print(f"  Avg Word Ratio      (Oromo/Amharic): {df['word_ratio'].mean():.2f}")
    
    print("\n📌 DOMAIN BREAKDOWN")
    domain_counts = df['Domain'].value_counts()
    for domain, count in domain_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {domain:18}: {count:>8,} rows ({pct:>5.1f}%)")
    print("="*80 + "\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_1_text_length_distributions(df):
    """Chart 1: Character & Word Count Distributions (Histograms & Boxplots)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Text Length & Word Count Distributions', fontsize=16, fontweight='bold')
    
    # Character Histogram
    ax = axes[0, 0]
    ax.hist(df['am_chars'], bins=30, alpha=0.6, label='Amharic', color='#3498db', edgecolor='black')
    ax.hist(df['or_chars'], bins=30, alpha=0.6, label='Afaan Oromo', color='#e74c3c', edgecolor='black')
    ax.set_title('Character Length Distribution', fontweight='bold')
    ax.set_xlabel('Character Count', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend()
    
    # Word Histogram
    ax = axes[0, 1]
    ax.hist(df['am_words'], bins=30, alpha=0.6, label='Amharic', color='#3498db', edgecolor='black')
    ax.hist(df['or_words'], bins=30, alpha=0.6, label='Afaan Oromo', color='#e74c3c', edgecolor='black')
    ax.set_title('Word Count Distribution', fontweight='bold')
    ax.set_xlabel('Word Count', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend()
    
    # Character Boxplot
    ax = axes[1, 0]
    bp1 = ax.boxplot([df['am_chars'], df['or_chars']], tick_labels=['Amharic', 'Oromo'], patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
    ax.set_title('Character Length Box Plot', fontweight='bold')
    ax.set_ylabel('Character Count', fontweight='bold')
    
    # Word Boxplot
    ax = axes[1, 1]
    bp2 = ax.boxplot([df['am_words'], df['or_words']], tick_labels=['Amharic', 'Oromo'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
    ax.set_title('Word Count Box Plot', fontweight='bold')
    ax.set_ylabel('Word Count', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_length_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Visualization: data/processed/1_length_distributions.png")
    plt.close()


def plot_2_domain_analysis(df):
    """Chart 2: Domain Share & Word Length Comparison Across Domains"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Domain Analysis', fontsize=16, fontweight='bold')
    
    domain_counts = df['Domain'].value_counts()
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Domain Counts
    ax = axes[0]
    ax.barh(domain_counts.index, domain_counts.values, color=colors[:len(domain_counts)], edgecolor='black')
    for i, (domain, count) in enumerate(domain_counts.items()):
        ax.text(count, i, f' {count:,} ({count/len(df)*100:.1f}%)', va='center', fontweight='bold')
    ax.set_title('Sentence Counts per Domain', fontweight='bold')
    ax.set_xlabel('Number of Parallel Pairs', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Avg Word Count by Domain
    ax = axes[1]
    domain_words = df.groupby('Domain')[['am_words', 'or_words']].mean()
    x = np.arange(len(domain_words))
    width = 0.35
    
    ax.bar(x - width/2, domain_words['am_words'], width, label='Amharic', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, domain_words['or_words'], width, label='Afaan Oromo', color='#e74c3c', edgecolor='black')
    ax.set_title('Average Word Count by Domain', fontweight='bold')
    ax.set_ylabel('Average Word Count', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_words.index, rotation=35, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_domain_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Visualization: data/processed/2_domain_analysis.png")
    plt.close()


def plot_3_parallel_correlation(df):
    """Chart 3: Amharic vs Oromo Correlation & Length Ratio Distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Parallel Alignment & Length Ratios', fontsize=16, fontweight='bold')
    
    # Word Alignment Scatter Plot
    ax = axes[0]
    sns.scatterplot(data=df, x='am_words', y='or_words', hue='Domain', alpha=0.6, ax=ax)
    ax.set_title('Word Count Alignment (Amharic vs Oromo)', fontweight='bold')
    ax.set_xlabel('Amharic Word Count', fontweight='bold')
    ax.set_ylabel('Afaan Oromo Word Count', fontweight='bold')
    
    # Length Ratio Distribution (Expansion factor)
    ax = axes[1]
    valid_ratios = df['char_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    sns.histplot(valid_ratios, bins=30, kde=True, color='#2ecc71', ax=ax, edgecolor='black')
    ax.axvline(valid_ratios.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Ratio ({valid_ratios.mean():.2f})')
    ax.set_title('Character Length Expansion Ratio (Oromo / Amharic)', fontweight='bold')
    ax.set_xlabel('Ratio', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_parallel_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Visualization: data/processed/3_parallel_correlation.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("RUNNING EDA ON CLASSIFIED DATASET")
    print("="*80)
    
    # 1. Load Data
    df = load_data()
    
    # 2. Compute Metrics
    df = calculate_metrics(df)
    
    # 3. Print Text Summary
    print_eda_summary(df)
    
    # 4. Generate & Save Charts
    print("📈 CREATING EDA VISUALIZATIONS...")
    plot_1_text_length_distributions(df)
    plot_2_domain_analysis(df)
    plot_3_parallel_correlation(df)
    
    print("\n" + "="*80)
    print("✓ EDA COMPLETE SUCCESSFULLY!")
    print("="*80)
    print("📊 Visualizations saved to: data/processed/")
    print("   1. 1_length_distributions.png  - Chars & Words distribution histograms and boxplots")
    print("   2. 2_domain_analysis.png        - Domain breakdown & average word count per domain")
    print("   3. 3_parallel_correlation.png    - Sentence length correlation & ratio analysis\n")

if __name__ == "__main__":
    main()