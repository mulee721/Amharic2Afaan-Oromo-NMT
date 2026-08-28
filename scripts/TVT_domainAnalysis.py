"""
═══════════════════════════════════════════════════════════════════════════════
PROFESSIONAL DOMAIN DISTRIBUTION FAIRNESS ANALYZER
Train/Validation/Test Dataset Fairness Assessment
═══════════════════════════════════════════════════════════════════════════════

Features:
  ✅ Load train/val/test datasets automatically
  ✅ Multiple statistical tests (Chi-Square, Kolmogorov-Smirnov, etc.)
  ✅ Professional visualizations
  ✅ Fairness assessment and recommendations
  ✅ Detailed HTML report generation
  ✅ Data quality checks
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp, entropy
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directory configuration
DATA_DIR = "data/processed"
if not os.path.exists(DATA_DIR) and os.path.exists(os.path.join("..", "data", "processed")):
    DATA_DIR = os.path.join("..", "data", "processed")

# Dataset files configuration
FILES = {
    "Train": ["train.csv"],
    "Validation": ["valid.csv"],
    "Test": ["test.csv"]
}

# Column name for domain/category
DOMAIN_COLUMN = "domain"  # Change if your column is different

# Output directory
OUTPUT_DIR = DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_dataset_file(file_names):
    """Find dataset file by checking multiple extensions"""
    for filename in file_names:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            return path
    return None

def load_dataset(path):
    """Load dataset with automatic format detection"""
    if path.endswith('.tsv'):
        return pd.read_csv(path, sep='\t', encoding='utf-8')
    elif path.endswith('.json'):
        return pd.read_json(path, lines=True, encoding='utf-8')
    else:
        return pd.read_csv(path, encoding='utf-8')

def load_datasets():
    """Load all datasets"""
    datasets = {}
    print(f"📁 Looking for datasets in: {os.path.abspath(DATA_DIR)}\n")
    
    for split_name, file_patterns in FILES.items():
        path = find_dataset_file(file_patterns)
        
        if path and os.path.exists(path):
            try:
                df = load_dataset(path)
                datasets[split_name] = df
                print(f"✅ {split_name:12} ({os.path.basename(path):20}): {len(df):>8,} rows")
            except Exception as e:
                print(f"❌ {split_name:12}: Error loading - {e}")
        else:
            print(f"⚠️  {split_name:12}: File not found")
    
    return datasets

# ============================================================================
# DOMAIN EXTRACTION & VALIDATION
# ============================================================================

def extract_domains(df, column_name):
    """Extract domains, handling various column names"""
    # Try exact match
    if column_name in df.columns:
        return df[column_name]
    
    # Try case-insensitive
    for col in df.columns:
        if col.lower() == column_name.lower():
            return df[col]
    
    # Try common alternatives
    alternatives = ['category', 'source', 'type', 'label', 'class', 'topic']
    for alt in alternatives:
        if alt in df.columns:
            print(f"   ℹ️  Using column '{alt}' instead of '{column_name}'")
            return df[alt]
        for col in df.columns:
            if col.lower() == alt:
                print(f"   ℹ️  Using column '{col}' instead of '{column_name}'")
                return df[col]
    
    return None

def validate_datasets(datasets):
    """Validate datasets and extract domains"""
    validated = {}
    
    for split_name, df in datasets.items():
        print(f"\n{'='*70}")
        print(f"Validating {split_name} Dataset")
        print(f"{'='*70}")
        
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        
        # Extract domains
        domains = extract_domains(df, DOMAIN_COLUMN)
        
        if domains is None:
            print(f"  ❌ ERROR: Could not find domain column!")
            print(f"     Available columns: {list(df.columns)}")
            continue
        
        # Check for missing values
        missing_count = domains.isna().sum()
        if missing_count > 0:
            print(f"  ⚠️  Warning: {missing_count} missing domain values ({missing_count/len(df)*100:.1f}%)")
            domains = domains.dropna()
        
        # Check for empty strings
        empty_count = (domains.astype(str) == '').sum()
        if empty_count > 0:
            print(f"  ⚠️  Warning: {empty_count} empty domain values")
        
        print(f"  ✅ Valid domains: {len(domains):,} rows")
        print(f"  📊 Unique domains: {domains.nunique()}")
        
        validated[split_name] = {
            'dataframe': df,
            'domains': domains,
            'domain_counts': domains.value_counts(),
            'valid_rows': len(domains)
        }
    
    return validated

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def calculate_distribution_metrics(validated_data):
    """Calculate comprehensive distribution metrics"""
    
    print(f"\n\n{'='*70}")
    print("DOMAIN DISTRIBUTION ANALYSIS")
    print(f"{'='*70}\n")
    
    # Create comparison table
    all_domains = set()
    for data in validated_data.values():
        all_domains.update(data['domains'].unique())
    
    all_domains = sorted(list(all_domains))
    
    comparison_table = pd.DataFrame(index=all_domains)
    
    for split_name, data in validated_data.items():
        counts = data['domain_counts']
        percents = (counts / len(data['domains']) * 100).round(2)
        
        comparison_table[f"{split_name} Count"] = counts
        comparison_table[f"{split_name} %"] = percents
    
    comparison_table = comparison_table.fillna(0)
    
    print(comparison_table.to_string())
    print(f"\n{'='*70}\n")
    
    return comparison_table

def chi_square_test(validated_data):
    """Perform Chi-Square Test of Independence"""
    
    print("🔬 STATISTICAL TESTS FOR FAIRNESS\n")
    print("─" * 70)
    print("1. CHI-SQUARE TEST (Independence Across Splits)")
    print("─" * 70)
    
    # Build contingency table
    contingency_df = pd.DataFrame()
    for split_name, data in validated_data.items():
        contingency_df[split_name] = data['domain_counts']
    
    contingency_df = contingency_df.fillna(0).astype(int)
    
    # Perform test
    chi2, p_value, dof, expected = chi2_contingency(contingency_df)
    
    print(f"Null Hypothesis: Domain distributions are INDEPENDENT across splits")
    print(f"Alternative: Domain distributions are DEPENDENT across splits\n")
    print(f"  Chi-Square Statistic: {chi2:.4f}")
    print(f"  P-Value: {p_value:.6f}")
    print(f"  Degrees of Freedom: {dof}")
    print(f"  Significance Level: α = 0.05\n")
    
    if p_value > 0.05:
        print(f"  ✅ RESULT: FAIR DISTRIBUTION (p > 0.05)")
        print(f"     Domains are fairly balanced across train/val/test splits.")
    else:
        print(f"  ⚠️  RESULT: UNFAIR DISTRIBUTION (p < 0.05)")
        print(f"     Domain distributions differ significantly across splits.")
        print(f"     Recommendation: Review split strategy, may need resampling.")
    
    return {
        'test': 'Chi-Square',
        'statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'fair': p_value > 0.05
    }

def calculate_distribution_entropy(validated_data):
    """Calculate entropy for each split"""
    
    print("\n─" * 70)
    print("2. DISTRIBUTION ENTROPY ANALYSIS")
    print("─" * 70)
    print("(Higher entropy = more uniform distribution)\n")
    
    entropy_results = {}
    
    for split_name, data in validated_data.items():
        # Calculate probability distribution
        probs = data['domains'].value_counts() / len(data['domains'])
        
        # Calculate entropy (normalized)
        ent = entropy(probs)
        max_ent = np.log(len(probs))  # Maximum possible entropy
        norm_ent = ent / max_ent if max_ent > 0 else 0
        
        entropy_results[split_name] = {
            'entropy': ent,
            'normalized_entropy': norm_ent,
            'num_domains': len(probs)
        }
        
        print(f"  {split_name:12} - Entropy: {ent:6.3f} | Normalized: {norm_ent:6.3f} | Domains: {len(probs)}")
    
    print("\n  Interpretation:")
    print("    0.0-0.3: Very skewed (one domain dominates)")
    print("    0.3-0.6: Moderately skewed")
    print("    0.6-0.9: Fairly balanced")
    print("    0.9-1.0: Highly uniform")
    
    return entropy_results

def calculate_kl_divergence(validated_data):
    """Calculate KL Divergence between distributions"""
    
    print("\n─" * 70)
    print("3. KL DIVERGENCE ANALYSIS (Distribution Similarity)")
    print("─" * 70)
    print("(Lower KL divergence = more similar distributions)\n")
    
    from scipy.spatial.distance import jensenshannon
    
    splits = list(validated_data.keys())
    all_domains = set()
    for data in validated_data.values():
        all_domains.update(data['domains'].unique())
    
    all_domains = sorted(list(all_domains))
    
    # Create probability vectors
    prob_vectors = {}
    for split_name, data in validated_data.items():
        probs = []
        for domain in all_domains:
            count = (data['domains'] == domain).sum()
            probs.append(count / len(data['domains']))
        prob_vectors[split_name] = np.array(probs)
    
    # Calculate pairwise KL divergence (using Jensen-Shannon symmetric divergence)
    print("  Jensen-Shannon Divergence (symmetric, 0=identical, 1=completely different):\n")
    
    for i, split1 in enumerate(splits):
        for split2 in splits[i+1:]:
            js_div = jensenshannon(prob_vectors[split1], prob_vectors[split2])
            print(f"    {split1} ↔ {split2}: {js_div:.4f}")
    
    print("\n  Interpretation:")
    print("    0.0-0.1: Very similar distributions ✅")
    print("    0.1-0.2: Similar distributions")
    print("    0.2-0.5: Moderately different")
    print("    0.5+:    Very different distributions ⚠️")

def chi_square_homogeneity(validated_data):
    """Chi-Square Test for Homogeneity of proportions"""
    
    print("\n─" * 70)
    print("4. CHI-SQUARE TEST FOR HOMOGENEITY")
    print("─" * 70)
    print("(Tests if domain proportions are same across splits)\n")
    
    # Build contingency table
    contingency_df = pd.DataFrame()
    for split_name, data in validated_data.items():
        contingency_df[split_name] = data['domain_counts']
    
    contingency_df = contingency_df.fillna(0).astype(int)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_df)
    
    print(f"  Chi-Square Statistic: {chi2:.4f}")
    print(f"  P-Value: {p_value:.6f}")
    print(f"  Degrees of Freedom: {dof}\n")
    
    if p_value > 0.05:
        print(f"  ✅ CONCLUSION: Proportions are HOMOGENEOUS (p > 0.05)")
        print(f"     Domain proportions are statistically similar across splits.")
    else:
        print(f"  ⚠️  CONCLUSION: Proportions are HETEROGENEOUS (p < 0.05)")
        print(f"     Domain proportions differ significantly.")

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(validated_data, comparison_table):
    """Create professional visualizations"""
    
    print("\n📊 Generating visualizations...\n")
    
    # Figure 1: Domain Distribution Comparison (Bar Chart)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Domain Distribution Analysis: Train/Validation/Test', fontsize=16, fontweight='bold')
    
    # Subplot 1: Stacked bar chart
    ax = axes[0, 0]
    comparison_table.filter(regex='Count').plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_title('Absolute Domain Counts', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_xlabel('Domain', fontweight='bold')
    ax.legend(title='Split')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot 2: Percentage comparison
    ax = axes[0, 1]
    comparison_table.filter(regex='%').plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_title('Domain Distribution Percentages', fontweight='bold', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xlabel('Domain', fontweight='bold')
    ax.legend(title='Split')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot 3: Pie charts
    ax = axes[1, 0]
    splits = [name for name in validated_data.keys()]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    sizes = [len(validated_data[split]['domains']) for split in splits]
    ax.pie(sizes, labels=splits, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Split Size Distribution', fontweight='bold', fontsize=12)
    
    # Subplot 4: Heatmap of proportions
    ax = axes[1, 1]
    prop_table = comparison_table.filter(regex='%').copy()
    prop_table = prop_table.iloc[:, 0::2]  # Keep only percentage columns
    prop_table.columns = [col.replace(' %', '') for col in prop_table.columns]
    
    sns.heatmap(prop_table.T, annot=True, fmt='.1f', cmap='YlGn', ax=ax, cbar_kws={'label': 'Percentage (%)'})
    ax.set_title('Domain Percentage Heatmap', fontweight='bold', fontsize=12)
    ax.set_xlabel('Domain', fontweight='bold')
    ax.set_ylabel('Split', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'domain_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_path}")
    plt.close()

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_html_report(validated_data, comparison_table, chi_square_result, entropy_results):
    """Generate professional HTML report"""
    
    print("\n📄 Generating HTML report...\n")
    
    # Prepare data
    table_html = comparison_table.to_html(classes='table table-striped')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Domain Distribution Fairness Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .metric {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                border-left: 4px solid #3498db;
            }}
            .result-pass {{
                background-color: #d5f4e6;
                border-left-color: #27ae60;
            }}
            .result-fail {{
                background-color: #fadbd8;
                border-left-color: #e74c3c;
            }}
            .result-warning {{
                background-color: #fef5e7;
                border-left-color: #f39c12;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th {{
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .icon-pass {{ color: #27ae60; font-weight: bold; }}
            .icon-fail {{ color: #e74c3c; font-weight: bold; }}
            .icon-warning {{ color: #f39c12; font-weight: bold; }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 12px;
                text-align: right;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Domain Distribution Fairness Analysis Report</h1>
            
            <h2>📊 Domain Distribution Summary</h2>
            {table_html}
            
            <h2>🔬 Statistical Test Results</h2>
            <div class="metric {'result-pass' if chi_square_result['fair'] else 'result-fail'}">
                <strong>Chi-Square Test of Independence:</strong><br>
                χ² = {chi_square_result['statistic']:.4f}, p-value = {chi_square_result['p_value']:.6f}<br>
                {'<span class="icon-pass">✅ FAIR DISTRIBUTION</span>' if chi_square_result['fair'] else '<span class="icon-fail">⚠️ UNFAIR DISTRIBUTION</span>'}
            </div>
            
            <h2>📈 Distribution Entropy</h2>
            <table>
                <tr>
                    <th>Split</th>
                    <th>Entropy</th>
                    <th>Normalized Entropy</th>
                    <th>Domains</th>
                    <th>Interpretation</th>
                </tr>
    """
    
    for split_name, metrics in entropy_results.items():
        ne = metrics['normalized_entropy']
        if ne > 0.9:
            interpretation = "Highly uniform ✅"
        elif ne > 0.6:
            interpretation = "Fairly balanced ✅"
        elif ne > 0.3:
            interpretation = "Moderately skewed ⚠️"
        else:
            interpretation = "Very skewed ❌"
        
        html_content += f"""
                <tr>
                    <td>{split_name}</td>
                    <td>{metrics['entropy']:.4f}</td>
                    <td>{metrics['normalized_entropy']:.4f}</td>
                    <td>{metrics['num_domains']}</td>
                    <td>{interpretation}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>✅ Recommendations</h2>
            <ul>
                <li>If p-value > 0.05: Domain distribution is fair across splits ✅</li>
                <li>If p-value ≤ 0.05: Consider stratified train/test split</li>
                <li>High entropy (>0.9): Good domain diversity</li>
                <li>Low entropy (<0.3): May need domain balancing</li>
            </ul>
            
            <div class="timestamp">
                Report generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
            </div>
        </div>
    </body>
    </html>
    """
    
    output_path = os.path.join(OUTPUT_DIR, 'domain_distribution_report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Saved HTML report: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PROFESSIONAL DOMAIN DISTRIBUTION FAIRNESS ANALYZER")
    print("="*70 + "\n")
    
    # Load datasets
    datasets = load_datasets()
    
    if not datasets:
        print("\n❌ ERROR: No datasets loaded. Check your data directory structure.")
        sys.exit(1)
    
    # Validate datasets
    validated_data = validate_datasets(datasets)
    
    if not validated_data:
        print("\n❌ ERROR: Could not extract domain information from datasets.")
        sys.exit(1)
    
    # Perform analysis
    comparison_table = calculate_distribution_metrics(validated_data)
    chi_square_result = chi_square_test(validated_data)
    entropy_results = calculate_distribution_entropy(validated_data)
    calculate_kl_divergence(validated_data)
    chi_square_homogeneity(validated_data)
    
    # Save comparison table
    csv_path = os.path.join(OUTPUT_DIR, 'domain_distribution_comparison.csv')
    comparison_table.to_csv(csv_path)
    print(f"\n✅ Saved comparison table: {csv_path}")
    
    # Generate visualizations
    create_visualizations(validated_data, comparison_table)
    
    # Generate report
    generate_html_report(validated_data, comparison_table, chi_square_result, entropy_results)
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE ✅")
    print("="*70)
    print(f"\nOutput files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("  - domain_distribution_comparison.csv")
    print("  - domain_distribution_analysis.png")
    print("  - domain_distribution_report.html")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
