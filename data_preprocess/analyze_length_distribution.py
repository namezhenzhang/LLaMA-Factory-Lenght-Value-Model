"""Analyze and visualize the length distribution of generated DAPO data.

This script strictly requires each sample to contain a valid
`meta_info.answer_token_length` field (positive number). If any
sample is missing this field or it is invalid, the script will
raise an error instead of silently skipping it.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def load_token_lengths(data_path: Path) -> List[int]:
    """Load answer token lengths from JSONL file."""
    lengths = []
    
    if not data_path.exists():
        print(f"Error: File {data_path} does not exist.")
        return lengths
    
    with data_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

            # Strictly require meta_info.answer_token_length
            meta_info = entry.get("meta_info")
            if meta_info is None or "answer_token_length" not in meta_info:
                raise ValueError(
                    f"Line {line_num}: missing required field 'meta_info.answer_token_length'."
                )

            token_length = meta_info.get("answer_token_length")
            if not isinstance(token_length, (int, float)) or token_length <= 0:
                raise ValueError(
                    f"Line {line_num}: invalid 'answer_token_length' value: {token_length!r}"
                )

            lengths.append(int(token_length))
    
    return lengths


def plot_distribution(
    lengths: List[int],
    output_path: Path,
    title: str = "Answer Token Length Distribution",
    bins: int = 50,
):
    """Create and save distribution plots."""
    if not lengths:
        print("No data to plot.")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Calculate statistics
    lengths_array = np.array(lengths)
    mean_length = np.mean(lengths_array)
    median_length = np.median(lengths_array)
    std_length = np.std(lengths_array)
    min_length = np.min(lengths_array)
    max_length = np.max(lengths_array)
    
    # 1. Histogram
    ax1 = axes[0, 0]
    n, bins_edges, patches = ax1.hist(lengths, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
    ax1.axvline(median_length, color='green', linestyle='--', linewidth=2, label=f'Median: {median_length:.1f}')
    ax1.set_xlabel('Token Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Histogram', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    bp = ax2.boxplot(lengths, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Token Length', fontsize=12)
    ax2.set_title('Box Plot', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_lengths = np.sort(lengths_array)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax3.plot(sorted_lengths, cumulative, linewidth=2, color='steelblue')
    ax3.axhline(50, color='green', linestyle='--', linewidth=1, alpha=0.7, label='50th percentile')
    ax3.axhline(90, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='90th percentile')
    ax3.axhline(95, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95th percentile')
    ax3.set_xlabel('Token Length', fontsize=12)
    ax3.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax3.set_title('Cumulative Distribution', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary (text)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(lengths_array, percentiles)
    
    stats_text = f"""
    Statistics Summary
    {'='*40}
    
    Total Samples:     {len(lengths):,}
    
    Mean:              {mean_length:.2f}
    Median:            {median_length:.2f}
    Std Dev:           {std_length:.2f}
    
    Min:               {min_length:,}
    Max:               {max_length:,}
    Range:             {max_length - min_length:,}
    
    Percentiles:
    """
    
    for p, v in zip(percentiles, percentile_values):
        stats_text += f"    {p}th:              {v:.0f}\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Adjust layout and save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plot saved to: {output_path}")
    
    # Also print statistics to console
    print("\n" + "="*50)
    print("Statistics Summary")
    print("="*50)
    print(f"Total Samples:     {len(lengths):,}")
    print(f"Mean:              {mean_length:.2f}")
    print(f"Median:            {median_length:.2f}")
    print(f"Std Dev:           {std_length:.2f}")
    print(f"Min:               {min_length:,}")
    print(f"Max:               {max_length:,}")
    print(f"Range:             {max_length - min_length:,}")
    print("\nPercentiles:")
    for p, v in zip(percentiles, percentile_values):
        print(f"  {p}th:              {v:.0f}")
    print("="*50 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution from generated DAPO data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the JSONL data file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the distribution plot (default: same directory as data file)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for histogram (default: 50)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Answer Token Length Distribution",
        help="Title for the plot",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    data_path = Path(args.data_path)
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = data_path.with_suffix('.png')
    
    print(f"Loading data from: {data_path}")
    
    # Load token lengths
    lengths = load_token_lengths(data_path)
    
    if not lengths:
        print("Error: No valid token lengths found in the data file.")
        return
    
    print(f"Loaded {len(lengths)} samples with token length information.")
    
    # Create and save plot
    plot_distribution(lengths, output_path, title=args.title, bins=args.bins)


if __name__ == "__main__":
    main()
