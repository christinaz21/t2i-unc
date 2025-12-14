#!/usr/bin/env python
import argparse
import pandas as pd
from pathlib import Path
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze experiment results by prompt specificity category"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory containing experiment results.",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/category_analysis",
        help="Output directory for analysis results.",
    )
    
    return parser.parse_args()


def load_exp1_results(results_dir: Path, category: str) -> pd.DataFrame:
    """Load Experiment 1 results for a category."""
    csv_path = results_dir / "exp1_generative" / category / f"exp1_generative_{category}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_exp3_results(results_dir: Path, category: str) -> pd.DataFrame:
    """Load Experiment 3 results for a category."""
    csv_path = results_dir / "exp3_aleatoric" / category / f"exp3_aleatoric_{category}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_exp4_results(results_dir: Path) -> pd.DataFrame:
    """Load Experiment 4 results."""
    csv_path = results_dir / "exp4_epistemic" / "exp4_epistemic.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def summarize_by_category(df: pd.DataFrame, metrics: list, category_col: str = "category") -> pd.DataFrame:
    """Compute summary statistics by category."""
    if df is None or len(df) == 0:
        return None
    
    summary = df.groupby(category_col)[metrics].agg(['mean', 'std', 'count']).round(4)
    return summary


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = ["abstract", "concrete", "moderate", "underspecified"]
    
    print("="*80)
    print("Analyzing Results by Prompt Specificity Category")
    print("="*80)
    
    # Experiment 1: Generative Uncertainty
    print("\nExperiment 1: Generative Uncertainty")
    print("-" * 80)
    exp1_summaries = []
    
    for category in categories:
        df = load_exp1_results(results_dir, category)
        if df is not None:
            metrics = [
                "clip_mean_similarity", "clip_variance_similarity",
                "lpips_mean_distance", "lpips_variance_distance",
                "latent_mean_cosine_similarity", "latent_variance_cosine_similarity"
            ]
            summary = summarize_by_category(df, metrics)
            if summary is not None:
                summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
                summary.index.name = 'category'
                summary = summary.reset_index()
                summary['category'] = category
                exp1_summaries.append(summary)
                print(f"  {category}: {len(df)} prompts")
    
    if exp1_summaries:
        exp1_combined = pd.concat(exp1_summaries, ignore_index=True)
        exp1_path = output_dir / "exp1_by_category.csv"
        exp1_combined.to_csv(exp1_path, index=False)
        print(f"  Saved: {exp1_path}")
    
    # Experiment 3: Aleatoric Uncertainty
    print("\nExperiment 3: Aleatoric Uncertainty")
    print("-" * 80)
    exp3_summaries = []
    
    for category in categories:
        df = load_exp3_results(results_dir, category)
        if df is not None:
            metrics = [
                "sim_prompt_image", "sim_caption_image", "sim_prompt_caption",
                "delta_image_sim", "aleatoric_prompt_caption"
            ]
            summary = summarize_by_category(df, metrics)
            if summary is not None:
                summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
                summary.index.name = 'category'
                summary = summary.reset_index()
                summary['category'] = category
                exp3_summaries.append(summary)
                print(f"  {category}: {len(df)} prompts")
    
    if exp3_summaries:
        exp3_combined = pd.concat(exp3_summaries, ignore_index=True)
        exp3_path = output_dir / "exp3_by_category.csv"
        exp3_combined.to_csv(exp3_path, index=False)
        print(f"  Saved: {exp3_path}")
    
    # Experiment 4: Epistemic Uncertainty
    print("\nExperiment 4: Epistemic Uncertainty")
    print("-" * 80)
    df_exp4 = load_exp4_results(results_dir)
    if df_exp4 is not None:
        metrics = ["mean_similarity", "variance_similarity"]
        summary = summarize_by_category(df_exp4, metrics)
        if summary is not None:
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            summary.index.name = 'category'
            summary = summary.reset_index()
            exp4_path = output_dir / "exp4_by_category.csv"
            summary.to_csv(exp4_path, index=False)
            print(f"  Total prompts: {len(df_exp4)}")
            print(f"  Saved: {exp4_path}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

