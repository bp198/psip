"""
Sprint 2 Demo: Pipeline Network Graph with Physics-Informed Vulnerability
==========================================================================

Generates a 20-node, 30-segment synthetic pipeline network calibrated to
PHMSA fleet statistics, attaches Monte Carlo P_f from the Sprint 1 engine,
and produces publication-ready network vulnerability visualizations.

Figures:
    fig11: Network topology map colored by P_f (vulnerability heatmap)
    fig12: Network property distributions (diameter, grade, seam type, P_f)

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import networkx as nx
from pathlib import Path

from src.zone_c.network.pipeline_graph import (
    PipelineNetwork, NodeType, SeamType,
)

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR / ".."
OUTPUT_DIR = PROJECT_DIR / "docs"
ETC_DIR = Path("/sessions/magical-peaceful-curie/mnt/etc")
FIGURES_DIR = ETC_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
})

# ===================================================================
# Step 1: Generate Synthetic Pipeline Network
# ===================================================================

print("=" * 70)
print("SPRINT 2: Pipeline Network Graph Model")
print("=" * 70)

net = PipelineNetwork("Gulf_Coast_Transmission_30seg")
net.generate_synthetic(
    n_nodes=20,
    n_segments=30,
    seed=42,
    region_bounds=(29.0, -96.0, 33.0, -88.0),  # Gulf Coast corridor
)

print(f"\n  Network generated: {net.n_nodes} nodes, {net.n_edges} edges")
print(f"  Total pipeline length: {net.total_length_km:.1f} km")

# Print node type counts
summary = net.summary()
print(f"  Node types: {summary['node_types']}")
print(f"  Connected: {summary['is_connected']}")

# ===================================================================
# Step 2: Attach Physics-Informed P_f Values
# ===================================================================

print("\n  Computing P_f for each segment (N=10,000 MC per segment)...")
net.attach_pf_values(n_simulations=10_000, seed=42)

summary = net.summary()
print(f"\n  P_f range: [{summary['P_f_range'][0]:.4f}, {summary['P_f_range'][1]:.4f}]")
print(f"  P_f mean:  {summary['P_f_mean']:.4f}")
print(f"  Seam types: {summary['seam_type_counts']}")
print(f"  Grades: {summary['grade_counts']}")

# Export edge data
edge_df = net.to_edge_dataframe()
node_df = net.to_node_dataframe()

print(f"\n  Edge data shape: {edge_df.shape}")
print(f"  Columns: {list(edge_df.columns)}")

# Print top-5 most vulnerable segments
edge_df_sorted = edge_df.sort_values("P_f", ascending=False)
print("\n  Top 5 most vulnerable segments:")
for _, row in edge_df_sorted.head(5).iterrows():
    print(f"    {row['segment_id']}: P_f={row['P_f']:.4f} | "
          f"D={row['diameter_mm']:.0f}mm | {row['seam_type']} | {row['grade']} | "
          f"FAT {row['fat_class']}")

print("\n  Bottom 5 (safest) segments:")
for _, row in edge_df_sorted.tail(5).iterrows():
    print(f"    {row['segment_id']}: P_f={row['P_f']:.4f} | "
          f"D={row['diameter_mm']:.0f}mm | {row['seam_type']} | {row['grade']} | "
          f"FAT {row['fat_class']}")


# ===================================================================
# Step 3: Figure 11 — Network Vulnerability Map
# ===================================================================

print("\n" + "=" * 70)
print("Generating Figure 11: Network Vulnerability Map")
print("=" * 70)

fig11, ax = plt.subplots(figsize=(14, 9))

G = net.graph

# Node positions from lat/lon
pos = {}
for n, d in G.nodes(data=True):
    if d.get("lon") is not None and d.get("lat") is not None:
        pos[n] = (d["lon"], d["lat"])

# --- Draw edges colored by P_f ---
edge_colors = []
edge_widths = []
for u, v, d in G.edges(data=True):
    pf = d.get("P_f", 0.5)
    edge_colors.append(pf)
    # Width proportional to diameter
    edge_widths.append(max(1.0, d.get("diameter_mm", 300) / 150))

edges = list(G.edges())
edge_collection = nx.draw_networkx_edges(
    G, pos, edgelist=edges, edge_color=edge_colors,
    edge_cmap=plt.cm.RdYlGn_r, edge_vmin=0.3, edge_vmax=0.95,
    width=edge_widths, alpha=0.85, arrows=True,
    arrowsize=8, arrowstyle="-|>",
    connectionstyle="arc3,rad=0.05",
    ax=ax,
)

# --- Draw nodes by type ---
node_type_config = {
    "source":     {"color": "#2ecc71", "marker": "s", "size": 200, "label": "Source"},
    "compressor": {"color": "#3498db", "marker": "^", "size": 180, "label": "Compressor"},
    "valve":      {"color": "#9b59b6", "marker": "d", "size": 120, "label": "Valve"},
    "junction":   {"color": "#95a5a6", "marker": "o", "size": 100, "label": "Junction"},
    "delivery":   {"color": "#e74c3c", "marker": "v", "size": 180, "label": "Delivery"},
    "storage":    {"color": "#f39c12", "marker": "H", "size": 160, "label": "Storage"},
}

for ntype, cfg in node_type_config.items():
    nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == ntype]
    if nodes_of_type:
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes_of_type,
            node_color=cfg["color"], node_shape=cfg["marker"],
            node_size=cfg["size"], edgecolors="black", linewidths=0.8,
            ax=ax,
        )

# Node labels
nx.draw_networkx_labels(G, pos, font_size=6, font_weight="bold", ax=ax)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                            norm=plt.Normalize(vmin=0.3, vmax=0.95))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Probability of Failure (P$_f$)", fontsize=11)

# Legend for node types
legend_handles = []
for ntype, cfg in node_type_config.items():
    nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == ntype]
    if nodes_of_type:
        legend_handles.append(
            mpatches.Patch(color=cfg["color"], label=f'{cfg["label"]} ({len(nodes_of_type)})')
        )
ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
          framealpha=0.9, edgecolor="gray")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    f"Pipeline Network Vulnerability Map — {net.name}\n"
    f"{net.n_nodes} nodes, {net.n_edges} segments, "
    f"{net.total_length_km:.0f} km total | "
    f"P$_f$ range: [{summary['P_f_range'][0]:.3f}, {summary['P_f_range'][1]:.3f}]",
    fontweight="bold",
)
ax.grid(True, alpha=0.2)
fig11.tight_layout()

for dest in [OUTPUT_DIR, FIGURES_DIR]:
    fig11.savefig(dest / "fig11_network_vulnerability_map.png", bbox_inches="tight")
plt.close(fig11)
print("  Saved fig11_network_vulnerability_map.png")


# ===================================================================
# Step 4: Figure 12 — Network Property Distributions
# ===================================================================

print("\nGenerating Figure 12: Network Property Distributions")

fig12, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: P_f distribution
ax = axes[0, 0]
pf_vals = edge_df["P_f"].values
ax.hist(pf_vals, bins=15, color="#e74c3c", edgecolor="black", linewidth=0.5, alpha=0.8)
ax.axvline(x=np.mean(pf_vals), color="navy", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(pf_vals):.3f}")
ax.axvline(x=np.median(pf_vals), color="green", linestyle=":", linewidth=1.5,
           label=f"Median = {np.median(pf_vals):.3f}")
ax.set_xlabel("P$_f$ (Failure Probability)")
ax.set_ylabel("Number of Segments")
ax.set_title("(a) P$_f$ Distribution Across Network")
ax.legend(fontsize=8)

# Panel B: Seam type breakdown with P_f overlay
ax = axes[0, 1]
seam_groups = edge_df.groupby("seam_type")["P_f"].agg(["mean", "count"]).sort_values("mean")
colors_seam = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(seam_groups)))
bars = ax.barh(range(len(seam_groups)), seam_groups["mean"],
               color=colors_seam, edgecolor="black", linewidth=0.5)
for i, (idx, row) in enumerate(seam_groups.iterrows()):
    ax.text(row["mean"] + 0.005, i,
            f'P$_f$={row["mean"]:.3f} (n={int(row["count"])})',
            va="center", fontsize=7.5)
ax.set_yticks(range(len(seam_groups)))
ax.set_yticklabels([s.replace("_", " ").title() for s in seam_groups.index], fontsize=8)
ax.set_xlabel("Mean P$_f$")
ax.set_title("(b) Mean P$_f$ by Seam Type")

# Panel C: Grade distribution
ax = axes[1, 0]
grade_groups = edge_df.groupby("grade")["P_f"].agg(["mean", "count"]).sort_values("mean")
colors_grade = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grade_groups)))
bars = ax.bar(range(len(grade_groups)), grade_groups["count"],
              color=colors_grade, edgecolor="black", linewidth=0.5)
ax2 = ax.twinx()
ax2.plot(range(len(grade_groups)), grade_groups["mean"], "ko-", markersize=6, linewidth=1.5)
ax2.set_ylabel("Mean P$_f$", color="black")
ax.set_xticks(range(len(grade_groups)))
ax.set_xticklabels(grade_groups.index, fontsize=9)
ax.set_xlabel("API 5L Grade")
ax.set_ylabel("Number of Segments")
ax.set_title("(c) Grade Distribution & Mean P$_f$")

# Panel D: Diameter vs P_f scatter
ax = axes[1, 1]
scatter = ax.scatter(
    edge_df["diameter_mm"], edge_df["P_f"],
    c=edge_df["P_f"], cmap="RdYlGn_r", vmin=0.3, vmax=0.95,
    s=edge_df["length_km"] * 0.8, alpha=0.7, edgecolors="black", linewidths=0.5,
)
ax.set_xlabel("Pipe Diameter (mm)")
ax.set_ylabel("P$_f$")
ax.set_title("(d) Diameter vs P$_f$ (size = segment length)")
plt.colorbar(scatter, ax=ax, shrink=0.8, label="P$_f$")

fig12.suptitle(
    "Pipeline Network Property & Vulnerability Analysis\n"
    f"{net.n_edges} segments | PHMSA-calibrated synthetic topology",
    fontweight="bold", fontsize=13,
)
fig12.tight_layout(rect=[0, 0, 1, 0.94])

for dest in [OUTPUT_DIR, FIGURES_DIR]:
    fig12.savefig(dest / "fig12_network_property_distributions.png", bbox_inches="tight")
plt.close(fig12)
print("  Saved fig12_network_property_distributions.png")


# ===================================================================
# Summary
# ===================================================================

print("\n" + "=" * 70)
print("SPRINT 2 NETWORK DEMO — COMPLETE")
print("=" * 70)
print(f"\n  Network: {net.name}")
print(f"  Nodes: {net.n_nodes} | Edges: {net.n_edges} | Length: {net.total_length_km:.0f} km")
print(f"  P_f range: [{summary['P_f_range'][0]:.4f}, {summary['P_f_range'][1]:.4f}]")
print(f"  P_f mean:  {summary['P_f_mean']:.4f}")
print(f"  Figures: fig11 (vulnerability map), fig12 (property distributions)")
print(f"  Game-theoretic payoff matrix: READY for Sprint 3")
