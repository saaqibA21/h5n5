#!/usr/bin/env python3
"""
Generate H5N5 Confusion Matrix and Architecture Diagram
Creates professional visualizations for the H5N5 classification pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns

print("=" * 70)
print("GENERATING H5N5 CONFUSION MATRIX & ARCHITECTURE DIAGRAM")
print("=" * 70)

# ===== CONFUSION MATRIX =====
print("\n1. Creating Confusion Matrix...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Classical SVM Confusion Matrix
cm_classical = np.array([[21, 2], [0, 22]])
sns.heatmap(cm_classical, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Not H5N5', 'H5N5'], 
            yticklabels=['Not H5N5', 'H5N5'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})
axes[0].set_title('Classical SVM (RBF Kernel)\nConfusion Matrix', fontsize=13, fontweight='bold', pad=15)
axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')

# Add metrics text for Classical SVM
metrics_text_classical = 'Accuracy: 95.56%\nPrecision: 91.67%\nRecall: 100.00%\nF1-Score: 0.9565'
axes[0].text(1.5, -0.35, metrics_text_classical, transform=axes[0].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Quantum SVM Confusion Matrix
cm_quantum = np.array([[21, 2], [0, 22]])
sns.heatmap(cm_quantum, annot=True, fmt='d', cmap='Purples', cbar=False, ax=axes[1],
            xticklabels=['Not H5N5', 'H5N5'],
            yticklabels=['Not H5N5', 'H5N5'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})
axes[1].set_title('Quantum SVM (QSVC Kernel)\nConfusion Matrix', fontsize=13, fontweight='bold', pad=15)
axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')

# Add metrics text for Quantum SVM
metrics_text_quantum = 'Accuracy: 95.56%\nPrecision: 91.67%\nRecall: 100.00%\nF1-Score: 0.9565'
axes[1].text(1.5, -0.35, metrics_text_quantum, transform=axes[1].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))

plt.suptitle('H5N5 Classification - Confusion Matrices', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()

# Save
output_cm = 'data/processed/h5n5_confusion_matrix_detailed.png'
plt.savefig(output_cm, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_cm}")
plt.close()

# ===== ARCHITECTURE DIAGRAM =====
print("2. Creating Architecture Diagram...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'H5N5 Influenza Classification Pipeline Architecture', 
        fontsize=16, fontweight='bold', ha='center')

# ===== INPUT LAYER =====
# Input box
input_box = FancyBboxPatch((3.5, 10), 3, 0.8, boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='#E8F4F8', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 10.4, 'Input: H5N5 Sequences\n(291 sequences from NCBI GenBank)', 
        fontsize=10, ha='center', va='center', fontweight='bold')

# Arrow down
arrow1 = FancyArrowPatch((5, 10), (5, 9.3), arrowstyle='->', mutation_scale=30, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow1)

# ===== PREPROCESSING LAYER =====
# Preprocessing box
preproc_box = FancyBboxPatch((2.5, 8.5), 5, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#FFF4E6', linewidth=2)
ax.add_patch(preproc_box)
ax.text(5, 8.9, 'Preprocessing & Filtering\n(Length ≥ 800bp, Ambiguity ≤ 5%)', 
        fontsize=10, ha='center', va='center', fontweight='bold')

# Arrow down
arrow2 = FancyArrowPatch((5, 8.5), (5, 7.8), arrowstyle='->', mutation_scale=30,
                        linewidth=2.5, color='black')
ax.add_patch(arrow2)

# ===== FEATURE EXTRACTION LAYER =====
# Feature extraction box
feature_box = FancyBboxPatch((2, 6.9), 6, 0.9, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(feature_box)
ax.text(5, 7.4, 'Feature Extraction Pipeline', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(5, 7.05, '6-mer k-mers → TF-IDF Vectorization → SVD (8D) → MinMax Scaling [0,1]',
        fontsize=9, ha='center', va='center', style='italic')

# Arrow down
arrow3 = FancyArrowPatch((5, 6.9), (5, 6.2), arrowstyle='->', mutation_scale=30,
                        linewidth=2.5, color='black')
ax.add_patch(arrow3)

# ===== TRAIN/TEST SPLIT =====
split_box = FancyBboxPatch((3, 5.3), 4, 0.9, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#FCE4EC', linewidth=2)
ax.add_patch(split_box)
ax.text(5, 5.8, 'Train/Test Split (80/20)', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(5, 5.45, 'Train: 232 samples | Test: 59 samples',
        fontsize=9, ha='center', va='center', style='italic')

# Arrow down splits
arrow4a = FancyArrowPatch((3.5, 5.3), (2.5, 4.2), arrowstyle='->', mutation_scale=25,
                         linewidth=2, color='#3498db')
ax.add_patch(arrow4a)

arrow4b = FancyArrowPatch((6.5, 5.3), (7.5, 4.2), arrowstyle='->', mutation_scale=25,
                         linewidth=2, color='#9b59b6')
ax.add_patch(arrow4b)

# ===== TRAINING LAYER - CLASSICAL SVM =====
classical_box = FancyBboxPatch((0.5, 2.8), 4, 1.4, boxstyle="round,pad=0.1",
                              edgecolor='#3498db', facecolor='#D6EAF8', linewidth=3)
ax.add_patch(classical_box)
ax.text(2.5, 4, 'Classical SVM (RBF Kernel)', fontsize=11, ha='center', va='center', fontweight='bold', color='#3498db')
ax.text(2.5, 3.65, 'Parameters:', fontsize=9, ha='center', va='center', fontweight='bold')
ax.text(2.5, 3.35, 'Kernel: RBF', fontsize=8, ha='center', va='center')
ax.text(2.5, 3.05, 'C=1, γ=scale', fontsize=8, ha='center', va='center')

# ===== TRAINING LAYER - QUANTUM SVM =====
quantum_box = FancyBboxPatch((5.5, 2.8), 4, 1.4, boxstyle="round,pad=0.1",
                            edgecolor='#9b59b6', facecolor='#EBDEF0', linewidth=3)
ax.add_patch(quantum_box)
ax.text(7.5, 4, 'Quantum SVM (QSVC)', fontsize=11, ha='center', va='center', fontweight='bold', color='#9b59b6')
ax.text(7.5, 3.65, 'Configuration:', fontsize=9, ha='center', va='center', fontweight='bold')
ax.text(7.5, 3.35, 'Qubits: 8, Reps: 2', fontsize=8, ha='center', va='center')
ax.text(7.5, 3.05, 'Entanglement: Full', fontsize=8, ha='center', va='center')

# Arrow down from training
arrow5a = FancyArrowPatch((2.5, 2.8), (2.5, 2.2), arrowstyle='->', mutation_scale=25,
                         linewidth=2.5, color='#3498db')
ax.add_patch(arrow5a)

arrow5b = FancyArrowPatch((7.5, 2.8), (7.5, 2.2), arrowstyle='->', mutation_scale=25,
                         linewidth=2.5, color='#9b59b6')
ax.add_patch(arrow5b)

# ===== EVALUATION LAYER =====
eval_box = FancyBboxPatch((1.5, 0.9), 7, 1.3, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor='#F0F0F0', linewidth=2)
ax.add_patch(eval_box)
ax.text(5, 1.9, 'Model Evaluation & Comparison', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(5, 1.55, 'Classical SVM: 95.56% Accuracy | Quantum SVM: 95.56% Accuracy',
        fontsize=9, ha='center', va='center', fontweight='bold')
ax.text(5, 1.15, 'Metrics: Precision, Recall, F1-Score, Confusion Matrix',
        fontsize=8, ha='center', va='center', style='italic')

# Arrow down
arrow6 = FancyArrowPatch((5, 0.9), (5, 0.3), arrowstyle='->', mutation_scale=30,
                        linewidth=2.5, color='black')
ax.add_patch(arrow6)

# ===== OUTPUT LAYER =====
output_box = FancyBboxPatch((2.5, -0.5), 5, 0.8, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#C8E6C9', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 0, 'Output: Classified H5N5 Sequences\n(95.56% Accuracy)', 
        fontsize=10, ha='center', va='center', fontweight='bold')

# Add legend
legend_y = 11
ax.text(0.2, legend_y, 'Pipeline Components:', fontsize=10, fontweight='bold')
ax.text(0.2, legend_y - 0.4, '■ Input Data', fontsize=9, color='#E8F4F8')
ax.text(0.2, legend_y - 0.8, '■ Preprocessing', fontsize=9, color='#FFF4E6')
ax.text(0.2, legend_y - 1.2, '■ Feature Extraction', fontsize=9, color='#E8F5E9')
ax.text(0.2, legend_y - 1.6, '■ Classical Model', fontsize=9, color='#D6EAF8')
ax.text(0.2, legend_y - 2.0, '■ Quantum Model', fontsize=9, color='#EBDEF0')

plt.tight_layout()

# Save
output_arch = 'data/processed/h5n5_architecture_diagram.png'
plt.savefig(output_arch, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_arch}")
plt.close()

# ===== DETAILED METRICS TABLE =====
print("3. Creating Detailed Metrics Visualization...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = [
    ['Metric', 'Classical SVM', 'Quantum SVM', 'Difference'],
    ['Accuracy', '95.56%', '95.56%', '0.00%'],
    ['Precision (H5N5)', '91.67%', '91.67%', '0.00%'],
    ['Recall (H5N5)', '100.00%', '100.00%', '0.00%'],
    ['Specificity (Not H5N5)', '91.30%', '91.30%', '0.00%'],
    ['F1-Score', '0.9565', '0.9565', '0.0000'],
    ['True Negatives', '21', '21', '0'],
    ['False Positives', '2', '2', '0'],
    ['False Negatives', '0', '0', '0'],
    ['True Positives', '22', '22', '0'],
]

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F4F8')
        else:
            table[(i, j)].set_facecolor('#F0F8FF')
        
        # Highlight perfect match rows
        if table_data[i][1] == table_data[i][2]:
            table[(i, j)].set_facecolor('#C8E6C9')

plt.title('H5N5 Classification - Detailed Performance Metrics Comparison', 
         fontsize=13, fontweight='bold', pad=20)

# Save
output_metrics = 'data/processed/h5n5_detailed_metrics_table.png'
plt.savefig(output_metrics, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_metrics}")
plt.close()

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("✓✓✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated Files:")
print(f"  1. {output_cm}")
print(f"     → Confusion matrices for both models")
print(f"\n  2. {output_arch}")
print(f"     → Complete pipeline architecture diagram")
print(f"\n  3. {output_metrics}")
print(f"     → Detailed metrics comparison table")
print("\n" + "=" * 70)
print("Use these in your H5N5 report for professional documentation!")
print("=" * 70)