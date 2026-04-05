#!/usr/bin/env python3
"""
Generate Clean Professional H5N5 Workflow Diagram
Vertical flow with proper spacing and formatting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

print("Generating Professional H5N5 Workflow Diagram...")

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Title
title = ax.text(5, 19.5, 'H5N5 Influenza Classification Workflow', 
                fontsize=20, fontweight='bold', ha='center')

# Color scheme
colors = {
    'input': '#2196F3',
    'process': '#FF9800', 
    'feature': '#4CAF50',
    'split': '#9C27B0',
    'model': '#F44336',
    'eval': '#00BCD4',
    'output': '#8BC34A'
}

text_colors = {
    'input': 'white',
    'process': 'white',
    'feature': 'white',
    'split': 'white',
    'model': 'white',
    'eval': 'white',
    'output': 'white'
}

y_current = 18.5
box_width = 8
box_height = 0.8

# ===== STEP 1: DATA INPUT =====
step_num = ax.text(0.5, y_current, '1', fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle='circle', facecolor=colors['input'], 
                            edgecolor='black', linewidth=2, pad=0.3), color='white')

box1 = FancyBboxPatch((1.2, y_current - 0.4), box_width - 0.4, box_height, 
                      boxstyle="round,pad=0.1", edgecolor='black', 
                      facecolor=colors['input'], linewidth=2.5, alpha=0.9)
ax.add_patch(box1)
ax.text(5.4, y_current, 'INPUT: H5N5 Sequences from NCBI GenBank', 
        fontsize=12, ha='center', va='center', fontweight='bold', color=text_colors['input'])
ax.text(5.4, y_current - 0.6, 'H5N5 Positive: 112 | Other H5Nx Negative: 179 | Total: 291 sequences', 
        fontsize=10, ha='center', va='center', color=text_colors['input'])

y_current -= 1.5

# Arrow
arrow = FancyArrowPatch((5, y_current + 1.1), (5, y_current + 0.5), 
                       arrowstyle='->', mutation_scale=35, linewidth=3, color='black')
ax.add_patch(arrow)

# ===== STEP 2: PREPROCESSING =====
step_num = ax.text(0.5, y_current, '2', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=colors['process'],
                            edgecolor='black', linewidth=2, pad=0.3), color='white')

box2 = FancyBboxPatch((1.2, y_current - 0.4), box_width - 0.4, box_height * 1.2,
                      boxstyle="round,pad=0.1", edgecolor='black',
                      facecolor=colors['process'], linewidth=2.5, alpha=0.9)
ax.add_patch(box2)
ax.text(5.4, y_current + 0.2, 'DATA PREPROCESSING & FILTERING', 
        fontsize=12, ha='center', va='center', fontweight='bold', color=text_colors['process'])
ax.text(5.4, y_current - 0.3, 'Normalize | Remove Short (<800bp) | Filter Ambiguous (>5%) | Deduplicate', 
        fontsize=10, ha='center', va='center', color=text_colors['process'], style='italic')

y_current -= 1.8

# Arrow
arrow = FancyArrowPatch((5, y_current + 1.3), (5, y_current + 0.5),
                       arrowstyle='->', mutation_scale=35, linewidth=3, color='black')
ax.add_patch(arrow)

# ===== STEP 3: FEATURE EXTRACTION =====
step_num = ax.text(0.5, y_current, '3', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=colors['feature'],
                            edgecolor='black', linewidth=2, pad=0.3), color='white')

box3 = FancyBboxPatch((1.2, y_current - 0.4), box_width - 0.4, box_height * 1.4,
                      boxstyle="round,pad=0.1", edgecolor='black',
                      facecolor=colors['feature'], linewidth=2.5, alpha=0.9)
ax.add_patch(box3)
ax.text(5.4, y_current + 0.35, 'FEATURE EXTRACTION PIPELINE', 
        fontsize=12, ha='center', va='center', fontweight='bold', color=text_colors['feature'])
ax.text(5.4, y_current - 0.15, '6-mer K-mers  →  TF-IDF Vectorization  →  SVD (8D)  →  MinMax Scaling [0,1]', 
        fontsize=9, ha='center', va='center', color=text_colors['feature'], style='italic')

y_current -= 2.0

# Arrow
arrow = FancyArrowPatch((5, y_current + 1.5), (5, y_current + 0.5),
                       arrowstyle='->', mutation_scale=35, linewidth=3, color='black')
ax.add_patch(arrow)

# ===== STEP 4: TRAIN/TEST SPLIT =====
step_num = ax.text(0.5, y_current, '4', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=colors['split'],
                            edgecolor='black', linewidth=2, pad=0.3), color='white')

box4 = FancyBboxPatch((1.2, y_current - 0.4), box_width - 0.4, box_height,
                      boxstyle="round,pad=0.1", edgecolor='black',
                      facecolor=colors['split'], linewidth=2.5, alpha=0.9)
ax.add_patch(box4)
ax.text(5.4, y_current, 'TRAIN/TEST SPLIT (80/20)', 
        fontsize=12, ha='center', va='center', fontweight='bold', color=text_colors['split'])
ax.text(5.4, y_current - 0.6, 'Training Set: 232 samples (80%)  |  Test Set: 59 samples (20%)', 
        fontsize=10, ha='center', va='center', color=text_colors['split'])

y_current -= 1.5

# Arrow splits to three paths
arrow_left = FancyArrowPatch((5, y_current + 1.1), (1.8, y_current + 0.5),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='#D32F2F')
ax.add_patch(arrow_left)

arrow_mid = FancyArrowPatch((5, y_current + 1.1), (5, y_current + 0.5),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='#388E3C')
ax.add_patch(arrow_mid)

arrow_right = FancyArrowPatch((5, y_current + 1.1), (8.2, y_current + 0.5),
                             arrowstyle='->', mutation_scale=25, linewidth=2, color='#7B1FA2')
ax.add_patch(arrow_right)

# ===== STEP 5A: CLASSICAL SVM (LEFT) =====
step_model_y = y_current

box5a = FancyBboxPatch((0.2, step_model_y - 0.4), 3, box_height * 1.3,
                       boxstyle="round,pad=0.1", edgecolor='#D32F2F', 
                       facecolor='#FFEBEE', linewidth=2, alpha=0.9)
ax.add_patch(box5a)

ax.text(0.4, step_model_y + 0.35, '5A', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='#D32F2F', edgecolor='black', 
                 linewidth=1, pad=0.2), color='white')

ax.text(1.7, step_model_y + 0.3, 'CLASSICAL SVM', 
        fontsize=10, ha='center', va='center', fontweight='bold', color='#D32F2F')
ax.text(1.7, step_model_y - 0.15, 'RBF Kernel | CV\nC=1, γ=scale', 
        fontsize=8, ha='center', va='center', color='#D32F2F', style='italic')

# ===== STEP 5C: RANDOM FOREST (MIDDLE) =====
box5c = FancyBboxPatch((3.4, step_model_y - 0.4), 3.2, box_height * 1.3,
                       boxstyle="round,pad=0.1", edgecolor='#388E3C',
                       facecolor='#E8F5E9', linewidth=2, alpha=0.9)
ax.add_patch(box5c)

ax.text(3.6, step_model_y + 0.35, '5C', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='#388E3C', edgecolor='black',
                 linewidth=1, pad=0.2), color='white')

ax.text(5, step_model_y + 0.3, 'RANDOM FOREST', 
        fontsize=10, ha='center', va='center', fontweight='bold', color='#388E3C')
ax.text(5, step_model_y - 0.15, 'Ensemble | Grid Search\nTuned Depth & Est.', 
        fontsize=8, ha='center', va='center', color='#388E3C', style='italic')

# ===== STEP 5B: QUANTUM SVM (RIGHT) =====
box5b = FancyBboxPatch((6.8, step_model_y - 0.4), 3, box_height * 1.3,
                       boxstyle="round,pad=0.1", edgecolor='#7B1FA2',
                       facecolor='#F3E5F5', linewidth=2, alpha=0.9)
ax.add_patch(box5b)

ax.text(7.0, step_model_y + 0.35, '5B', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='#7B1FA2', edgecolor='black',
                 linewidth=1, pad=0.2), color='white')

ax.text(8.3, step_model_y + 0.3, 'QUANTUM SVM', 
        fontsize=10, ha='center', va='center', fontweight='bold', color='#7B1FA2')
ax.text(8.3, step_model_y - 0.15, 'ZZ Map | 8 Qubits\nFidelity Kernel', 
        fontsize=8, ha='center', va='center', color='#7B1FA2', style='italic')

y_current -= 1.8

# Arrows from all models
arrow_left2 = FancyArrowPatch((1.7, step_model_y - 0.5), (4.5, y_current + 0.5),
                             arrowstyle='->', mutation_scale=25, linewidth=2, color='#D32F2F')
ax.add_patch(arrow_left2)

arrow_mid2 = FancyArrowPatch((5, step_model_y - 0.5), (5, y_current + 0.5),
                             arrowstyle='->', mutation_scale=25, linewidth=2, color='#388E3C')
ax.add_patch(arrow_mid2)

arrow_right2 = FancyArrowPatch((8.3, step_model_y - 0.5), (5.5, y_current + 0.5),
                              arrowstyle='->', mutation_scale=25, linewidth=2, color='#7B1FA2')
ax.add_patch(arrow_right2)

# ===== STEP 6: EVALUATION =====
step_num = ax.text(0.5, y_current, '6', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=colors['eval'],
                            edgecolor='black', linewidth=2, pad=0.3), color='white')

box6 = FancyBboxPatch((1.2, y_current - 0.4), box_width - 0.4, box_height * 1.2,
                      boxstyle="round,pad=0.1", edgecolor='black',
                      facecolor=colors['eval'], linewidth=2.5, alpha=0.9)
ax.add_patch(box6)
ax.text(5.4, y_current + 0.2, 'MODEL EVALUATION', 
        fontsize=12, ha='center', va='center', fontweight='bold', color=text_colors['eval'])
ax.text(5.4, y_current - 0.3, 'Test on 59 Unseen Sequences | Calculate Accuracy, Precision, Recall, F1', 
        fontsize=10, ha='center', va='center', color=text_colors['eval'], style='italic')

y_current -= 1.8

# Arrow
arrow = FancyArrowPatch((5, y_current + 1.3), (5, y_current + 0.5),
                       arrowstyle='->', mutation_scale=35, linewidth=3, color='black')
ax.add_patch(arrow)

# ===== STEP 7: RESULTS =====
step_num = ax.text(0.5, y_current, '7', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=colors['output'],
                            edgecolor='black', linewidth=2, pad=0.3), color='white')

box7 = FancyBboxPatch((1.2, y_current - 0.4), box_width - 0.4, box_height * 1.5,
                      boxstyle="round,pad=0.1", edgecolor='#1B5E20',
                      facecolor=colors['output'], linewidth=3, alpha=0.9)
ax.add_patch(box7)
ax.text(5.4, y_current + 0.4, 'RESULTS & METRICS', 
        fontsize=13, ha='center', va='center', fontweight='bold', color=text_colors['output'])
ax.text(5.4, y_current + 0.05, 'Classical SVM: 95.56% Accuracy | Quantum SVM: 95.56% Accuracy', 
        fontsize=10, ha='center', va='center', color=text_colors['output'], fontweight='bold')
ax.text(5.4, y_current - 0.35, 'Random Forest: 96.67% Accuracy | F1-Score: 0.96', 
        fontsize=10, ha='center', va='center', color=text_colors['output'], fontweight='bold')

y_current -= 2.0

# Arrow
arrow = FancyArrowPatch((5, y_current + 1.5), (5, y_current + 0.5),
                       arrowstyle='->', mutation_scale=35, linewidth=3, color='black')
ax.add_patch(arrow)

# ===== FINAL OUTPUT =====
box_final = FancyBboxPatch((0.8, y_current - 0.5), box_width + 0.4, box_height * 1.2,
                           boxstyle="round,pad=0.1", edgecolor='#1B5E20',
                           facecolor='#C8E6C9', linewidth=3)
ax.add_patch(box_final)
ax.text(5.4, y_current + 0.25, '✓ H5N5 CLASSIFICATION COMPLETE', 
        fontsize=13, ha='center', va='center', fontweight='bold', color='#1B5E20')
ax.text(5.4, y_current - 0.25, 'Professional Report with Visualizations, Metrics & References', 
        fontsize=10, ha='center', va='center', color='#1B5E20')

# Add summary box on right
summary_y = 9
summary_box = FancyBboxPatch((0.2, summary_y - 3.5), 9.6, 3.3,
                            boxstyle="round,pad=0.15", edgecolor='black',
                            facecolor='#F5F5F5', linewidth=1.5, linestyle='--', alpha=0.7)
ax.add_patch(summary_box)

summary_text = [
    'WORKFLOW SUMMARY:',
    '• Data: 291 H5N5 sequences (112 positive, 179 negative)',
    '• Features: 8-dimensional vectors from 6-mer k-mers',
    '• Models: Classical SVM, Quantum SVM, Random Forest',
    '• Split: 80% train (232), 20% test (59)',
    '• Performance: High accuracy across all models'
]

text_y = summary_y - 0.3
for i, line in enumerate(summary_text):
    if i == 0:
        ax.text(0.5, text_y, line, fontsize=11, fontweight='bold', color='black')
    else:
        ax.text(0.5, text_y, line, fontsize=10, color='#333333')
    text_y -= 0.5

plt.tight_layout()

# Save
output_path = 'data/processed/h5n5_workflow_professional.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {output_path}")
plt.close()

print("\n" + "=" * 70)
print("✓✓✓ PROFESSIONAL WORKFLOW DIAGRAM CREATED!")
print("=" * 70)
print(f"\nFile: {output_path}")
print("\n7-Step Workflow:")
print("  1️⃣  Input: NCBI GenBank H5N5 sequences (291 total)")
print("  2️⃣  Preprocessing: Filter, normalize, deduplicate")
print("  3️⃣  Feature Extraction: K-mers → TF-IDF → SVD → Scaling")
print("  4️⃣  Train/Test Split: 80/20 (232/59 samples)")
print("  5️⃣  Model Training: SVM (Classical/Quantum) & Random Forest")
print("  6️⃣  Evaluation: Test on 59 unseen sequences")
print("  7️⃣  Results: Comparative analysis of all models")
print("\n" + "=" * 70)