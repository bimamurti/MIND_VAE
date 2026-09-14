import matplotlib.pyplot as plt
import numpy as np

# Data from the table
#locations = ["Duri Morning", "Duri Evening", "vredeburg", "Platform1", "Paisley"]
#locations=["Platform1","vredeburg","Duri Morning","Paisley","Duri Evening"]
#locations=["data120b2","data90d7","data90g3","dataMOT212"]
locations=["data90g3","data90d7","dataMOT212","data120b2"]

# series = {
#         "Full Independent": [0.0562, 0.1108, 0.0835, 0.0851, 0.0912],
#         "Independent L": [0.0673, 0.1351, 0.0853, 0.0790, 0.0986],
#         "Zero Shot CL": [0.0620, 0.1179, 0.0834, 0.0828, 0.1620],
#         "Exp. Rep": [0.0643, 0.0737, 0.0817, 0.0740, 0.0987],
#         "BCSR": [0.0595581122, 0.0973506495, 0.0790458024, 0.0833500530, 0.0974313021],
#         "REL": [0.0595308505, 0.0975626707, 0.0812275708, 0.0832946301, 0.0861186191],
#         "Ensemble (ours)": [0.0626305938, 0.0748060346, 0.0745644942, 0.0772858411, 0.0777183920],
# }
# series = {
#         "Full Independent": [0.024, 0.0308, 0.032, 0.0316, 0.0307],
#         "Independent L": [0.0251, 0.0328, 0.032, 0.0317, 0.0307],
#         "Zero Shot CL": [0.0253, 0.0285, 0.0309, 0.0304, 0.031],
#         "Exp. Rep": [0.0257, 0.0272, 0.032, 0.0304, 0.0283],
#         "BCSR": [0.0252, 0.0273, 0.0356, 0.0307, 0.0321],
#         "REL": [0.0252, 0.0282, 0.0334, 0.0316, 0.0303],
#         "Ensemble (ours)": [0.0255, 0.0289, 0.0356, 0.0297, 0.0323],
# }


# series = {
#         "Full Independent": [2.6173145771, 3.0350239277, 3.0790567398, 3.3671443462],
#         "Independent CL": [2.6620268822, 3.0840647221, 3.1118333340, 5.4642646273],
#         "Zero Shot CL": [2.6805717945, 3.0706140995, 2.9208569527, 3.2266752720],
#         "Exp. Rep": [2.6138665676, 3.0965647697, 2.9746172428, 3.3098218440],
#         "BCSR": [2.6172375679, 3.0752627850, 3.0033094883, 3.2216486931],
#         "REL": [2.6133966446, 3.0979669090, 2.9732694626, 3.3124962440],
#         "Ensemble (ours)": [2.5729269981, 2.6461000443, 2.6339721680, 2.9353082180],
# }
series = {
        "Full Independent": [3.8602015972, 4.0659465790, 3.0790567398, 4.0112726245],
        "Independent L": [4.0975899696, 4.0959249829, 5.0664525032, 5.4642646273],
        "Zero Shot CL": [4.0992760658, 4.0888838768, 4.7127118111, 4.1172580092],
        "Exp. Rep": [3.6696074009, 3.8705649763, 4.6031723022, 4.0027151108],
        "BCSR": [3.6663341522, 3.9679858685, 4.5250229883, 3.9860596657],
        "REL": [3.6592056751, 3.8721470838, 4.6019053459, 3.9927656651],
        "Ensemble (ours)": [4.0113468170, 3.9837472439, 4.1989803314, 4.0975379944],
}
output_path = 'results_comparison3.png'
# Define colors and markers programmatically based on series keys
color_list = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#A23B72", "#F18F01", "#2E86AB"]
marker_list = ["o", "s", "^", "D", "v", "P", "X"]

colors = dict(zip(series.keys(), color_list))
markers = dict(zip(series.keys(), marker_list))

# Calculate dynamic y-axis limit: max value + 10%
max_value = max(max(values) for values in list(series.values()) + list(series.values()))
y_limit = max_value * 1.20

# Create figure and axis
fig, ax = plt.subplots(figsize=(5.5, 7))

x_pos = np.arange(len(locations)) * 0.5
line_width = 2.2
marker_size = 7

for label, values in series.items():
    ax.plot(
        x_pos,
        values,
        marker=markers[label],
        linewidth=line_width,
        markersize=marker_size,
        label=label,
        color=colors[label],
        linestyle='-',
    )

# Customize x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(locations, fontsize=11)
ax.set_xlabel('Trained Scenes', fontsize=12, fontweight='bold')
#ax.set_xlim(0, 2)
ax.margins(x=0.02)
# Customize y-axis
ax.set_ylabel('Error Metric (Pixels)', fontsize=12, fontweight='bold')
ax.set_ylim(1, y_limit)
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
ax.legend(fontsize=11, loc='upper left', framealpha=0.9,ncol=2)

# Title
plt.title('Performance every stage/task', fontsize=14, fontweight='bold', pad=20)

# Tight layout
plt.tight_layout()

# Save the figure

plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"Graph saved to {output_path}")

# Show the plot
plt.show()

# Optional: Create a second version with different styling
fig2, ax2 = plt.subplots(figsize=(12, 7))

x = np.arange(len(locations))
width = 0.12

offsets = np.linspace(-width * 3, width * 3, len(series))
bars = []
for offset, (label, values) in zip(offsets, series.items()):
        bars.append(
                ax2.bar(x + offset, values, width, label=label, color=colors[label], alpha=0.85)
        )

# Customize axes
ax2.set_xlabel('Dataset Location', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error Metric', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(locations, fontsize=11)
ax2.legend(fontsize=10, ncol=2)
ax2.set_ylim(0, y_limit)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
def add_value_labels(bars):
        for bar_container in bars:
                for bar in bar_container:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                         f'{height:.4f}', ha='center', va='bottom', fontsize=8)

add_value_labels(bars)

plt.title('Performance Comparison Across Different Datasets (Bar Chart)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save the bar chart
output_path_bar = 'results_comparison_bar.png'
plt.savefig(output_path_bar, dpi=600, bbox_inches='tight')
print(f"Bar chart saved to {output_path_bar}")

plt.show()

# Print statistics
print("\n" + "=" * 60)
print("PERFORMANCE STATISTICS")
print("=" * 60)
print(f"{'Method':<20} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
print("-" * 60)

for method_name, values in series.items():
        print(
                f"{method_name:<20} {np.mean(values):<10.4f} {np.std(values):<10.4f} "
                f"{np.min(values):<10.4f} {np.max(values):<10.4f}"
        )

print("\n" + "=" * 60)
print("BEST PERFORMING METHOD PER DATASET")
print("=" * 60)
for i, loc in enumerate(locations):
        methods = [(label, values[i]) for label, values in series.items()]
        best = min(methods, key=lambda x: x[1])
        print(f"{loc:<20} {best[0]:<20} ({best[1]:.4f})")
