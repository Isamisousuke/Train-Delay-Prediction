"""Historical analysis for the finalized 23-train dataset."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "01_Data" / "train_delay_23_trains.csv"
GRAPH_DIR = ROOT / "04_Graphs"
RESULT_DIR = ROOT / "01_Data"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_FILE)
    required = ["Train_Number", "Station_Name", "Average_Delay(min)",
        "Right Time (0-15 min's)", "Slight Delay (15-60 min's)",
        "Significant Delay (>1 Hour)", "Cancelled/Unknown",
        "Cumulative_Distance_km", "Distance_between_stations_km"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df

def run_analysis():
    df = load_data()
    category_columns = ["Right Time (0-15 min's)", "Slight Delay (15-60 min's)",
        "Significant Delay (>1 Hour)", "Cancelled/Unknown"]
    summary = df["Average_Delay(min)"].describe().round(2)
    category_means = df[category_columns].mean().round(2)
    summary.to_csv(RESULT_DIR / "historical_delay_summary.csv", header=["value"])
    category_means.to_csv(RESULT_DIR / "average_delay_categories.csv", header=["mean_percent"])
    plt.figure(figsize=(9, 5))
    plt.hist(df["Average_Delay(min)"].dropna(), bins=25, edgecolor="black")
    plt.xlabel("Historical average delay (minutes)")
    plt.ylabel("Number of station records")
    plt.title("Historical Station Delay Distribution — 23 Matched Trains")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "delay_distribution.png", dpi=180)
    plt.close()
    route_means = df.groupby("Train_Number", as_index=False)["Average_Delay(min)"].mean().sort_values("Average_Delay(min)", ascending=False)
    route_means.to_csv(RESULT_DIR / "historical_delay_by_train.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.bar(route_means["Train_Number"].astype(str), route_means["Average_Delay(min)"])
    plt.xlabel("Train number")
    plt.ylabel("Mean station historical average delay (minutes)")
    plt.title("Historical Delay Comparison by Train — 23 Matched Trains")
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "route_comparison.png", dpi=180)
    plt.close()
    labels = ["Right Time", "Slight Delay", "Significant Delay", "Cancelled/Unknown"]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, category_means.values)
    plt.ylabel("Mean reported percentage")
    plt.title("Historical Station Delay Category Distribution")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "delay_category_distribution.png", dpi=180)
    plt.close()
    print("HISTORICAL ANALYSIS COMPLETE")
    print(f"Matched trains: {df['Train_Number'].nunique()}")
    print(f"Station records with matched distances: {len(df)}")
    print("\nHistorical average-delay statistics (minutes):")
    print(summary.to_string())
    print("\nMean category percentages:")
    print(category_means.to_string())
    print(f"\nGraphs saved to: {GRAPH_DIR}")
    return df

if __name__ == "__main__":
    run_analysis()
