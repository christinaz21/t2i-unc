#!/usr/bin/env python
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize experiment results by prompt specificity category"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory containing experiment results.",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations",
        help="Output directory for visualization plots.",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots.",
    )
    
    return parser.parse_args()


def load_exp1_data(results_dir: Path, category: str) -> pd.DataFrame:
    """Load Experiment 1 results for a category."""
    csv_path = results_dir / "exp1_generative" / category / f"exp1_generative_{category}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['category'] = category
        return df
    return None


def load_exp3_data(results_dir: Path, category: str) -> pd.DataFrame:
    """Load Experiment 3 results for a category."""
    csv_path = results_dir / "exp3_aleatoric" / category / f"exp3_aleatoric_{category}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['category'] = category
        return df
    return None


def load_exp4_data(results_dir: Path) -> pd.DataFrame:
    """Load Experiment 4 results."""
    csv_path = results_dir / "exp4_epistemic" / "exp4_epistemic.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        # Fix typo in category name if present
        if 'category' in df.columns:
            df['category'] = df['category'].astype(str).str.replace('undersspecified', 'underspecified', regex=False)
        return df
    return None


def plot_exp1_generative(results_dir: Path, output_dir: Path, format: str):
    """Visualize Experiment 1: Generative Uncertainty."""
    print("Creating Experiment 1 visualizations...")
    
    categories = ["abstract", "concrete", "moderate", "underspecified"]
    all_data = []
    
    for category in categories:
        df = load_exp1_data(results_dir, category)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("  No Experiment 1 data found.")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Define category order for consistent plotting
    category_order = ["abstract", "underspecified", "moderate", "concrete"]
    df_all['category'] = pd.Categorical(df_all['category'], categories=category_order, ordered=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 1: Generative Uncertainty by Prompt Specificity', fontsize=14, fontweight='bold')
    
    # 1. CLIP Mean Similarity
    ax1 = axes[0, 0]
    sns.boxplot(data=df_all, x='category', y='clip_mean_similarity', ax=ax1, palette='Set2')
    ax1.set_title('CLIP Mean Similarity (Higher = More Consistent)')
    ax1.set_xlabel('Prompt Specificity')
    ax1.set_ylabel('Mean Similarity')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. CLIP Variance Similarity
    ax2 = axes[0, 1]
    sns.boxplot(data=df_all, x='category', y='clip_variance_similarity', ax=ax2, palette='Set2')
    ax2.set_title('CLIP Variance Similarity (Higher = More Variable)')
    ax2.set_xlabel('Prompt Specificity')
    ax2.set_ylabel('Variance')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. LPIPS Mean Distance
    ax3 = axes[1, 0]
    sns.boxplot(data=df_all, x='category', y='lpips_mean_distance', ax=ax3, palette='Set2')
    ax3.set_title('LPIPS Mean Distance (Higher = More Different)')
    ax3.set_xlabel('Prompt Specificity')
    ax3.set_ylabel('Mean Distance')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. LPIPS Variance Distance
    ax4 = axes[1, 1]
    sns.boxplot(data=df_all, x='category', y='lpips_variance_distance', ax=ax4, palette='Set2')
    ax4.set_title('LPIPS Variance Distance')
    ax4.set_xlabel('Prompt Specificity')
    ax4.set_ylabel('Variance')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / f"exp1_generative_by_category.{format}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Summary statistics table
    summary = df_all.groupby('category').agg({
        'clip_mean_similarity': ['mean', 'std'],
        'clip_variance_similarity': ['mean', 'std'],
        'lpips_mean_distance': ['mean', 'std'],
    }).round(4)
    summary_path = output_dir / "exp1_summary_stats.csv"
    summary.to_csv(summary_path)
    print(f"  Saved summary: {summary_path}")


def plot_exp3_aleatoric(results_dir: Path, output_dir: Path, format: str):
    """Visualize Experiment 3: Aleatoric Uncertainty."""
    print("Creating Experiment 3 visualizations...")
    
    categories = ["abstract", "concrete", "moderate", "underspecified"]
    all_data = []
    
    for category in categories:
        df = load_exp3_data(results_dir, category)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("  No Experiment 3 data found.")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Define category order
    category_order = ["abstract", "underspecified", "moderate", "concrete"]
    df_all['category'] = pd.Categorical(df_all['category'], categories=category_order, ordered=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 2: Aleatoric Uncertainty by Prompt Specificity', fontsize=14, fontweight='bold')
    
    # 1. Prompt-Image Similarity
    ax1 = axes[0, 0]
    sns.boxplot(data=df_all, x='category', y='sim_prompt_image', ax=ax1, palette='Set2')
    ax1.set_title('Prompt-Image Similarity')
    ax1.set_xlabel('Prompt Specificity')
    ax1.set_ylabel('Similarity')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Caption-Image Similarity
    ax2 = axes[0, 1]
    sns.boxplot(data=df_all, x='category', y='sim_caption_image', ax=ax2, palette='Set2')
    ax2.set_title('Caption-Image Similarity')
    ax2.set_xlabel('Prompt Specificity')
    ax2.set_ylabel('Similarity')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Prompt-Caption Similarity
    ax3 = axes[1, 0]
    sns.boxplot(data=df_all, x='category', y='sim_prompt_caption', ax=ax3, palette='Set2')
    ax3.set_title('Prompt-Caption Similarity (Lower = More Aleatoric Uncertainty)')
    ax3.set_xlabel('Prompt Specificity')
    ax3.set_ylabel('Similarity')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Aleatoric Prompt-Caption (1 - similarity)
    ax4 = axes[1, 1]
    if 'aleatoric_prompt_caption' in df_all.columns:
        sns.boxplot(data=df_all, x='category', y='aleatoric_prompt_caption', ax=ax4, palette='Set2')
        ax4.set_title('Aleatoric Uncertainty (1 - Prompt-Caption Sim)')
        ax4.set_xlabel('Prompt Specificity')
        ax4.set_ylabel('Uncertainty Score')
    else:
        # Calculate it
        df_all['aleatoric'] = 1 - df_all['sim_prompt_caption']
        sns.boxplot(data=df_all, x='category', y='aleatoric', ax=ax4, palette='Set2')
        ax4.set_title('Aleatoric Uncertainty (1 - Prompt-Caption Sim)')
        ax4.set_xlabel('Prompt Specificity')
        ax4.set_ylabel('Uncertainty Score')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / f"exp3_aleatoric_by_category.{format}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Summary statistics
    summary = df_all.groupby('category').agg({
        'sim_prompt_image': ['mean', 'std'],
        'sim_caption_image': ['mean', 'std'],
        'sim_prompt_caption': ['mean', 'std'],
    }).round(4)
    summary_path = output_dir / "exp3_summary_stats.csv"
    summary.to_csv(summary_path)
    print(f"  Saved summary: {summary_path}")


def plot_exp4_epistemic(results_dir: Path, output_dir: Path, format: str):
    """Visualize Experiment 4: Epistemic Uncertainty."""
    print("Creating Experiment 4 visualizations...")
    
    df = load_exp4_data(results_dir)
    if df is None or len(df) == 0:
        print("  No Experiment 4 data found.")
        return
    
    # Check if category column exists
    if 'category' not in df.columns:
        print("  Warning: 'category' column not found in Experiment 4 data.")
        print(f"  Available columns: {list(df.columns)}")
        print("  Skipping Experiment 4 visualizations.")
        return
    
    # Define category order
    category_order = ["abstract", "underspecified", "moderate", "concrete"]
    df['category'] = pd.Categorical(df['category'], categories=category_order, ordered=True)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Experiment 3: Epistemic Uncertainty (Cross-Model) by Prompt Specificity', 
                 fontsize=14, fontweight='bold')
    
    # 1. Mean Similarity (across models)
    ax1 = axes[0]
    sns.boxplot(data=df, x='category', y='mean_similarity', ax=ax1, palette='Set2')
    ax1.set_title('Mean Cross-Model Similarity (Higher = Models Agree More)')
    ax1.set_xlabel('Prompt Specificity')
    ax1.set_ylabel('Mean Similarity')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Variance Similarity (epistemic uncertainty)
    ax2 = axes[1]
    sns.boxplot(data=df, x='category', y='variance_similarity', ax=ax2, palette='Set2')
    ax2.set_title('Variance in Cross-Model Similarity (Higher = More Epistemic Uncertainty)')
    ax2.set_xlabel('Prompt Specificity')
    ax2.set_ylabel('Variance')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / f"exp4_epistemic_by_category.{format}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Summary statistics
    summary = df.groupby('category').agg({
        'mean_similarity': ['mean', 'std'],
        'variance_similarity': ['mean', 'std'],
    }).round(4)
    summary_path = output_dir / "exp4_summary_stats.csv"
    summary.to_csv(summary_path)
    print(f"  Saved summary: {summary_path}")


def plot_exp1_histograms(results_dir: Path, output_dir: Path, format: str):
    """Create histogram distributions for Experiment 1 metrics."""
    print("Creating Experiment 1 histogram distributions...")
    
    categories = ["abstract", "underspecified", "moderate", "concrete"]
    all_data = []
    
    for category in categories:
        df = load_exp1_data(results_dir, category)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("  No Experiment 1 data found.")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Create histogram plots for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 1: Distribution of Generative Uncertainty Metrics by Prompt Specificity', 
                 fontsize=14, fontweight='bold')
    
    # 1. CLIP Mean Similarity
    ax1 = axes[0, 0]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax1.hist(cat_data['clip_mean_similarity'], bins=30, alpha=0.6, label=category, density=True)
    ax1.set_title('CLIP Mean Similarity Distribution')
    ax1.set_xlabel('Mean Similarity')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CLIP Variance Similarity
    ax2 = axes[0, 1]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax2.hist(cat_data['clip_variance_similarity'], bins=30, alpha=0.6, label=category, density=True)
    ax2.set_title('CLIP Variance Similarity Distribution')
    ax2.set_xlabel('Variance')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. LPIPS Mean Distance
    ax3 = axes[1, 0]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax3.hist(cat_data['lpips_mean_distance'], bins=30, alpha=0.6, label=category, density=True)
    ax3.set_title('LPIPS Mean Distance Distribution')
    ax3.set_xlabel('Mean Distance')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. LPIPS Variance Distance
    ax4 = axes[1, 1]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax4.hist(cat_data['lpips_variance_distance'], bins=30, alpha=0.6, label=category, density=True)
    ax4.set_title('LPIPS Variance Distance Distribution')
    ax4.set_xlabel('Variance')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"exp1_histograms_by_category.{format}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_exp3_histograms(results_dir: Path, output_dir: Path, format: str):
    """Create histogram distributions for Experiment 3 metrics."""
    print("Creating Experiment 3 histogram distributions...")
    
    categories = ["abstract", "underspecified", "moderate", "concrete"]
    all_data = []
    
    for category in categories:
        df = load_exp3_data(results_dir, category)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("  No Experiment 3 data found.")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Calculate aleatoric uncertainty if not present
    if 'aleatoric_prompt_caption' not in df_all.columns:
        df_all['aleatoric_prompt_caption'] = 1 - df_all['sim_prompt_caption']
    
    # Create histogram plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 2: Distribution of Aleatoric Uncertainty Metrics by Prompt Specificity', 
                 fontsize=14, fontweight='bold')
    
    # 1. Prompt-Image Similarity
    ax1 = axes[0, 0]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax1.hist(cat_data['sim_prompt_image'], bins=30, alpha=0.6, label=category, density=True)
    ax1.set_title('Prompt-Image Similarity Distribution')
    ax1.set_xlabel('Similarity')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Caption-Image Similarity
    ax2 = axes[0, 1]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax2.hist(cat_data['sim_caption_image'], bins=30, alpha=0.6, label=category, density=True)
    ax2.set_title('Caption-Image Similarity Distribution')
    ax2.set_xlabel('Similarity')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Prompt-Caption Similarity
    ax3 = axes[1, 0]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax3.hist(cat_data['sim_prompt_caption'], bins=30, alpha=0.6, label=category, density=True)
    ax3.set_title('Prompt-Caption Similarity Distribution')
    ax3.set_xlabel('Similarity')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Aleatoric Uncertainty
    ax4 = axes[1, 1]
    for category in categories:
        cat_data = df_all[df_all['category'] == category]
        if len(cat_data) > 0:
            ax4.hist(cat_data['aleatoric_prompt_caption'], bins=30, alpha=0.6, label=category, density=True)
    ax4.set_title('Aleatoric Uncertainty Distribution (1 - Prompt-Caption Sim)')
    ax4.set_xlabel('Uncertainty Score')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"exp3_histograms_by_category.{format}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_exp4_histograms(results_dir: Path, output_dir: Path, format: str):
    """Create histogram distributions for Experiment 4 metrics."""
    print("Creating Experiment 4 histogram distributions...")
    
    df = load_exp4_data(results_dir)
    if df is None or len(df) == 0:
        print("  No Experiment 4 data found.")
        return
    
    # Check if category column exists
    if 'category' not in df.columns:
        print("  Warning: 'category' column not found in Experiment 4 data.")
        print("  Skipping Experiment 4 histogram visualizations.")
        return
    
    categories = ["abstract", "underspecified", "moderate", "concrete"]
    
    # Create histogram plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Experiment 3: Distribution of Epistemic Uncertainty Metrics by Prompt Specificity', 
                 fontsize=14, fontweight='bold')
    
    # 1. Mean Similarity
    ax1 = axes[0]
    for category in categories:
        cat_data = df[df['category'] == category]
        if len(cat_data) > 0:
            ax1.hist(cat_data['mean_similarity'], bins=30, alpha=0.6, label=category, density=True)
    ax1.set_title('Mean Cross-Model Similarity Distribution')
    ax1.set_xlabel('Mean Similarity')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Variance Similarity (Epistemic Uncertainty)
    ax2 = axes[1]
    for category in categories:
        cat_data = df[df['category'] == category]
        if len(cat_data) > 0:
            ax2.hist(cat_data['variance_similarity'], bins=30, alpha=0.6, label=category, density=True)
    ax2.set_title('Variance in Cross-Model Similarity Distribution (Epistemic Uncertainty)')
    ax2.set_xlabel('Variance')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"exp4_histograms_by_category.{format}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_comparison_across_experiments(results_dir: Path, output_dir: Path, format: str):
    """Create comparison plots across all three experiments."""
    print("Creating cross-experiment comparison visualizations...")
    
    # Load all data
    categories = ["abstract", "concrete", "moderate", "underspecified"]
    category_order = ["abstract", "underspecified", "moderate", "concrete"]
    
    # Experiment 1: Use CLIP variance as uncertainty metric
    exp1_data = []
    for cat in categories:
        df = load_exp1_data(results_dir, cat)
        if df is not None:
            exp1_data.append(df)
    if exp1_data:
        df_exp1 = pd.concat(exp1_data, ignore_index=True)
        df_exp1['uncertainty_metric'] = df_exp1['clip_variance_similarity']
        df_exp1['experiment'] = 'Generative'
    else:
        df_exp1 = None
    
    # Experiment 3: Use aleatoric uncertainty
    exp3_data = []
    for cat in categories:
        df = load_exp3_data(results_dir, cat)
        if df is not None:
            exp3_data.append(df)
    if exp3_data:
        df_exp3 = pd.concat(exp3_data, ignore_index=True)
        if 'aleatoric_prompt_caption' in df_exp3.columns:
            df_exp3['uncertainty_metric'] = df_exp3['aleatoric_prompt_caption']
        else:
            df_exp3['uncertainty_metric'] = 1 - df_exp3['sim_prompt_caption']
        df_exp3['experiment'] = 'Aleatoric'
    else:
        df_exp3 = None
    
    # Experiment 4: Use variance similarity
    df_exp4 = load_exp4_data(results_dir)
    if df_exp4 is not None and len(df_exp4) > 0:
        if 'category' not in df_exp4.columns:
            print("  Warning: 'category' column missing in Experiment 4 data for comparison plot.")
            df_exp4 = None
        else:
            df_exp4['uncertainty_metric'] = df_exp4['variance_similarity']
            df_exp4['experiment'] = 'Epistemic'
    else:
        df_exp4 = None
    
    # Combine all experiments
    all_experiments = []
    for df in [df_exp1, df_exp3, df_exp4]:
        if df is not None:
            all_experiments.append(df[['category', 'uncertainty_metric', 'experiment']])
    
    if not all_experiments:
        print("  No data available for comparison.")
        return
    
    df_combined = pd.concat(all_experiments, ignore_index=True)
    df_combined['category'] = pd.Categorical(df_combined['category'], categories=category_order, ordered=True)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_combined, x='category', y='uncertainty_metric', hue='experiment', 
                ax=ax, palette='muted')
    ax.set_title('Uncertainty Metrics Comparison Across Experiments by Prompt Specificity', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Prompt Specificity', fontsize=12)
    ax.set_ylabel('Uncertainty Score', fontsize=12)
    ax.legend(title='Experiment Type', title_fontsize=11, fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / f"comparison_all_experiments.{format}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Visualizing Experiment Results by Prompt Specificity")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create box plot visualizations for each experiment
    plot_exp1_generative(results_dir, output_dir, args.format)
    plot_exp3_aleatoric(results_dir, output_dir, args.format)
    plot_exp4_epistemic(results_dir, output_dir, args.format)
    plot_comparison_across_experiments(results_dir, output_dir, args.format)
    
    # Create histogram distributions for each experiment
    plot_exp1_histograms(results_dir, output_dir, args.format)
    plot_exp3_histograms(results_dir, output_dir, args.format)
    plot_exp4_histograms(results_dir, output_dir, args.format)
    
    print("\n" + "="*80)
    print("Visualization Complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

