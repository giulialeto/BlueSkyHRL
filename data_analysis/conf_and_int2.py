import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_analysis.opensky_tools.conflict_detection import get_conflict_data
from bluesky.tools.aero import ft, kts

# ---------- CONFIG ----------
TLOOK = 500  # s
ALT_CUTOFF = 1415  # m
R_SEP = 5556  # m
SAVE = False
CACHE_FILE = "data_analysis/data/conf_int.pkl"  # <-- cache location
FORCE_RECOMPUTE = False # True = generate new data, False = use cache if available
CUTOFF_DISTANCE = 250

FILES = [
    "output/synthetic/15ac_1/flight_output.csv",
    "output/synthetic/35ac_1/flight_output.csv",
    "output/synthetic/65ac_1/flight_output.csv",
    "output/synthetic/15ac_1_SA/flight_output.csv",
    "output/synthetic/35ac_1_SA/flight_output.csv",
    "output/synthetic/65ac_1_SA/flight_output.csv",
    "output/synthetic/15ac_1_direct/flight_output.csv",
    "output/synthetic/35ac_1_direct/flight_output.csv",
    "output/synthetic/65ac_1_direct/flight_output.csv",
]
FILES = [
    "output/jan_2024/flight_output.csv",
    "output/march_2024/flight_output.csv",
    "output/july_2024/flight_output.csv",
    "output/jan_2024_SA/flight_output.csv",
    "output/march_2024_SA/flight_output.csv",
    "output/july_2024_SA/flight_output.csv",
    "output/jan_2024_direct/flight_output.csv",
    "output/march_2024_direct/flight_output.csv",
    "output/july_2024_direct/flight_output.csv",
]

TRAFFIC_LEVELS = ["Jan", "Mar", "Jul"] * 3
METHODS = ["MA"] * 3 + ["SA"] * 3 + ["direct"] * 3

CHUNK_SIZE = 10000
N_TIMESTEPS = 100_000_000

def compute_results(force_recompute=False):
    """Compute conflict/intrusion results and cache them."""
    if os.path.exists(CACHE_FILE) and not force_recompute:
        print(f"Loading cached results from {CACHE_FILE}")
        return pd.read_pickle(CACHE_FILE)

    results = []

    for file, level, method in zip(FILES, TRAFFIC_LEVELS, METHODS):
        print(f"Processing {file} ({method}, {level})")
        chunks = pd.read_csv(file, chunksize=CHUNK_SIZE)
        total_rows = total_conflicts = total_conflict_time = total_intrusions = total_intrusion_time = 0

        for chunk in chunks:
            rows_needed = N_TIMESTEPS - total_rows
            if rows_needed < 0:
                break
            total_rows += len(chunk)

            chunk = chunk.rename(columns={
                "time": "timestamp",
                "lat": "latitude",
                "lon": "longitude",
                "heading": "track",
            }).drop(columns=["geoaltitude"])

            chunk = chunk.assign(
                altitude=chunk.baroaltitude / ft,
                vertical_rate=chunk.vertrate / ft * 60,
                groundspeed=chunk.velocity / kts,
            )

            summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(
                chunk[:rows_needed],
                r=R_SEP,
                save=SAVE,
                tlook=TLOOK,
                alt_cutoff=ALT_CUTOFF,
                cutoff_dist=CUTOFF_DISTANCE
            )

            total_conflicts += len(conflict_counts)
            total_conflict_time += conflict_counts["conflicts"].sum()
            total_intrusions += len(intrusion_counts)
            total_intrusion_time += intrusion_counts["intrusions"].sum()

        results.append({
            "Traffic Density": level,
            "Method": method,
            "Conflicts": total_conflicts,
            "Intrusions": total_intrusions,
            "Conflict Time": total_conflict_time,
            "Intrusion Time": total_intrusion_time
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    df.to_pickle(CACHE_FILE)
    print(f"Results saved to {CACHE_FILE}")
    return df

def plot_conflicts_intrusions(df):
    linestyle_map = {"MA": "-", "SA": "--", "direct": ":"}
    fig, ax1 = plt.subplots(figsize=(8, 6))
    color_conf, color_intr = "tab:blue", "tab:red"
    ax2 = ax1.twinx()

    for method in df["Method"].unique():
        subset = df[df["Method"] == method]
        ax1.plot(subset["Traffic Density"], subset["Conflicts"],
                 color=color_conf, linestyle=linestyle_map[method],
                 label=f"Conflicts ({method})")
        ax2.plot(subset["Traffic Density"], subset["Intrusions"],
                 color=color_intr, linestyle=linestyle_map[method],
                 label=f"Intrusions ({method})")

    ax1.set_xlabel("Traffic Density")
    ax1.set_ylabel("Conflicts", color=color_conf)
    ax2.set_ylabel("Intrusions", color=color_intr)
    ax1.tick_params(axis="y", labelcolor=color_conf)
    ax2.tick_params(axis="y", labelcolor=color_intr)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=1)

    # ax2.set_yscale('log')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("Conflicts and Intrusions by Traffic Density & Method")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()

def plot_conflict_intrusion_time(df):
    linestyle_map = {"MA": "-", "SA": "--", "direct": ":"}
    fig, ax1 = plt.subplots(figsize=(8, 6))
    color_conf, color_intr = "tab:blue", "tab:red"
    ax2 = ax1.twinx()

    for method in df["Method"].unique():
        subset = df[df["Method"] == method]
        ax1.plot(subset["Traffic Density"], subset["Conflict Time"],
                 color=color_conf, linestyle=linestyle_map[method],
                 label=f"Conflict Time ({method})")
        ax2.plot(subset["Traffic Density"], subset["Intrusion Time"],
                 color=color_intr, linestyle=linestyle_map[method],
                 label=f"Intrusion Time ({method})")

    ax1.set_xlabel("Traffic Density")
    ax1.set_ylabel("Conflict Time", color=color_conf)
    ax2.set_ylabel("Intrusion Time", color=color_intr)
    ax1.tick_params(axis="y", labelcolor=color_conf)
    ax2.tick_params(axis="y", labelcolor=color_intr)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=1)

    # ax2.set_yscale('log')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("Time in Conflict and Intrusion by Traffic Density & Method")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()

df_results = compute_results(force_recompute=FORCE_RECOMPUTE)
plot_conflicts_intrusions(df_results)
plot_conflict_intrusion_time(df_results)
