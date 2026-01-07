#!/usr/bin/env python3
"""
Generate academic-style graphs from enhanced evaluation results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_metrics(data):
    """Extract metrics from all samples."""
    phase1_clinical = []
    phase2_clinical = []
    phase1_relevancy = []
    phase2_relevancy = []
    phase2_faithfulness = []
    contextual_recall = []
    contextual_precision = []
    retrieval_scores = []

    for result in data['results']:
        p1 = result['phase1_metrics']
        p2 = result['phase2_metrics']
        ret = result['retrieval_metrics']

        phase1_clinical.append(p1['clinical_correctness'])
        phase2_clinical.append(p2['clinical_correctness'])
        phase1_relevancy.append(p1['answer_relevancy'])
        phase2_relevancy.append(p2['answer_relevancy'])
        phase2_faithfulness.append(p2['faithfulness'])
        contextual_recall.append(ret['contextual_recall'])
        contextual_precision.append(ret['contextual_precision'])
        retrieval_scores.append(ret['avg_retrieval_score'])

    return {
        'phase1_clinical': phase1_clinical,
        'phase2_clinical': phase2_clinical,
        'phase1_relevancy': phase1_relevancy,
        'phase2_relevancy': phase2_relevancy,
        'phase2_faithfulness': phase2_faithfulness,
        'contextual_recall': contextual_recall,
        'contextual_precision': contextual_precision,
        'retrieval_scores': retrieval_scores,
    }

def generate_graphs(data, output_dir):
    """Generate all academic graphs."""
    metrics = extract_metrics(data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate aggregates
    p1_clinical_avg = np.mean(metrics['phase1_clinical'])
    p2_clinical_avg = np.mean(metrics['phase2_clinical'])
    p1_relevancy_avg = np.mean(metrics['phase1_relevancy'])
    p2_relevancy_avg = np.mean(metrics['phase2_relevancy'])
    faithfulness_avg = np.mean(metrics['phase2_faithfulness'])
    recall_avg = np.mean(metrics['contextual_recall'])
    precision_avg = np.mean(metrics['contextual_precision'])

    # Figure 1: RAG Lift Bar Chart (Main Result)
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    metrics_names = ['Clinical\nCorrectness', 'Answer\nRelevancy']
    phase1_values = [p1_clinical_avg * 100, p1_relevancy_avg * 100]
    phase2_values = [p2_clinical_avg * 100, p2_relevancy_avg * 100]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, phase1_values, width, label='Without RAG', color='#e74c3c', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, phase2_values, width, label='With RAG', color='#2ecc71', edgecolor='black', linewidth=1.2)

    ax1.set_ylabel('Score (%)', fontweight='bold')
    ax1.set_title('TPN-GPT-20B: RAG Improvement on Clinical Q&A\n(n=20 samples)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 100)

    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add RAG Lift annotation
    lift = ((p2_clinical_avg - p1_clinical_avg) / p1_clinical_avg) * 100
    ax1.annotate(f'+{lift:.1f}% RAG Lift', xy=(0, p2_clinical_avg * 100), xytext=(0.5, 75),
                fontsize=14, fontweight='bold', color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

    plt.tight_layout()
    fig1.savefig(output_dir / 'rag_lift_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {output_dir / 'rag_lift_comparison.png'}")

    # Figure 2: Clinical Correctness by Sample
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    samples = range(1, len(metrics['phase1_clinical']) + 1)

    ax2.plot(samples, [x * 100 for x in metrics['phase1_clinical']], 'o-',
             color='#e74c3c', linewidth=2, markersize=8, label='Without RAG')
    ax2.plot(samples, [x * 100 for x in metrics['phase2_clinical']], 's-',
             color='#2ecc71', linewidth=2, markersize=8, label='With RAG')

    ax2.set_xlabel('Sample Number', fontweight='bold')
    ax2.set_ylabel('Clinical Correctness (%)', fontweight='bold')
    ax2.set_title('Clinical Correctness Scores by Sample\nPhase 1 (No RAG) vs Phase 2 (With RAG)', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(-5, 105)
    ax2.set_xticks(samples)

    plt.tight_layout()
    fig2.savefig(output_dir / 'clinical_correctness_by_sample.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {output_dir / 'clinical_correctness_by_sample.png'}")

    # Figure 3: Retrieval Quality Metrics
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    ret_metrics = ['Contextual\nRecall', 'Contextual\nPrecision', 'Faithfulness']
    ret_values = [recall_avg * 100, precision_avg * 100, faithfulness_avg * 100]
    colors = ['#3498db', '#9b59b6', '#f39c12']

    bars = ax3.bar(ret_metrics, ret_values, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Score (%)', fontweight='bold')
    ax3.set_title('Retrieval and Generation Quality Metrics', fontweight='bold')
    ax3.set_ylim(0, 100)

    for bar, val in zip(bars, ret_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig3.savefig(output_dir / 'retrieval_quality.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {output_dir / 'retrieval_quality.png'}")

    # Figure 4: Summary Dashboard
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top Left: RAG Lift
    ax = axes[0, 0]
    metrics_names = ['Clinical Correctness', 'Answer Relevancy']
    lifts = [
        ((p2_clinical_avg - p1_clinical_avg) / p1_clinical_avg) * 100,
        ((p2_relevancy_avg - p1_relevancy_avg) / p1_relevancy_avg) * 100
    ]
    colors = ['#2ecc71' if l > 0 else '#e74c3c' for l in lifts]
    ax.barh(metrics_names, lifts, color=colors, edgecolor='black', linewidth=1.2)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('RAG Lift (%)', fontweight='bold')
    ax.set_title('RAG Performance Improvement', fontweight='bold')
    for i, (val, name) in enumerate(zip(lifts, metrics_names)):
        ax.text(val + 2, i, f'+{val:.1f}%' if val > 0 else f'{val:.1f}%',
                va='center', fontsize=11, fontweight='bold')

    # Top Right: Phase Comparison
    ax = axes[0, 1]
    categories = ['Clinical\nCorrectness', 'Answer\nRelevancy', 'Faithfulness']
    phase1 = [p1_clinical_avg * 100, p1_relevancy_avg * 100, 0]
    phase2 = [p2_clinical_avg * 100, p2_relevancy_avg * 100, faithfulness_avg * 100]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, phase1, width, label='Phase 1 (No RAG)', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, phase2, width, label='Phase 2 (With RAG)', color='#2ecc71', alpha=0.8)
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 100)

    # Bottom Left: Retrieval Quality
    ax = axes[1, 0]
    sizes = [precision_avg * 100, (1 - precision_avg) * 100]
    colors_pie = ['#2ecc71', '#e74c3c']
    ax.pie(sizes, labels=[f'Relevant\n{precision_avg*100:.1f}%', f'Not Relevant\n{(1-precision_avg)*100:.1f}%'],
           colors=colors_pie, autopct='', startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Contextual Precision\n(Retrieved Document Relevance)', fontweight='bold')

    # Bottom Right: Summary Stats
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    EVALUATION SUMMARY
    ==================

    Samples Evaluated: {len(metrics['phase1_clinical'])}

    KEY RESULTS:

    Clinical Correctness:
      Without RAG: {p1_clinical_avg*100:.1f}%
      With RAG:    {p2_clinical_avg*100:.1f}%
      RAG Lift:    +{((p2_clinical_avg - p1_clinical_avg) / p1_clinical_avg) * 100:.1f}%

    Faithfulness: {faithfulness_avg*100:.1f}%

    Retrieval Quality:
      Precision: {precision_avg*100:.1f}%
      Recall:    {recall_avg*100:.1f}%
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig4.savefig(output_dir / 'evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {output_dir / 'evaluation_dashboard.png'}")

    print(f"\nAll graphs saved to: {output_dir}")

if __name__ == "__main__":
    results_path = Path(__file__).parent / "enhanced_results.json"
    output_dir = Path(__file__).parent / "graphs"

    print("Loading evaluation results...")
    data = load_results(results_path)

    print("Generating academic-style graphs...")
    generate_graphs(data, output_dir)
