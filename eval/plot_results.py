import matplotlib.pyplot as plt
import seaborn as sns

# Data
models = ['Base Qwen2.5-Coder-7B', 'Code-Autopsy (Fine-tuned)']
bleu_scores = [10.63, 70.03]

# Set style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Create bar plot
colors = ['#ff9999', '#66b3ff']
bars = plt.bar(models, bleu_scores, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# Add title and labels
plt.title('Code Quality Improvement Post Fine-Tuning (20 Samples)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('sacreBLEU Score (Higher is Better)', fontsize=14, labelpad=15)
plt.ylim(0, 100)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Customize axes
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=11)
plt.grid(axis='x')
sns.despine()

# Add a delta annotation arrow
plt.annotate(
    '  +59.40 BLEU\n(+558% Gain)',
    xy=(0.5, 40), 
    xycoords='data',
    xytext=(0.5, 40), textcoords='data',
    fontsize=14, fontweight='bold', color='green',
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="green", lw=2),
    ha='center'
)

# Save plot
plt.tight_layout()
plt.savefig('eval/results/bleu_comparison.png', dpi=300, bbox_inches='tight')
print("Saved plot to eval/results/bleu_comparison.png")
