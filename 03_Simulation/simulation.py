from pathlib import Path
import random
from datetime import date
import io

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "01_Data" / "train_delay_23_trains.csv"

LOW_THRESHOLD = 32
MEDIAN_THRESHOLD = 61
UPPER_THRESHOLD = 102.5

RANK = {
    "Right Time": 0,
    "Slight Delay": 1,
    "Significant Delay": 2,
}

CATEGORIES = [
    "Right Time (0-15 min's)",
    "Slight Delay (15-60 min's)",
    "Significant Delay (>1 Hour)",
    "Cancelled/Unknown",
]

DISPLAY_CATEGORIES = [
    "Right Time",
    "Slight Delay",
    "Significant Delay",
    "Cancelled/Unknown",
]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    required = [
        "Train_Number", "Station_Name", "Cumulative_Distance_km",
        "Average_Delay(min)", *CATEGORIES,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    for column in ["Train_Number", "Cumulative_Distance_km", "Average_Delay(min)", *CATEGORIES]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Train_Number", "Cumulative_Distance_km", "Average_Delay(min)"]).copy()
    df["Train_Number"] = df["Train_Number"].astype(int)
    df = df.sort_values(["Train_Number", "Cumulative_Distance_km"]).reset_index(drop=True)
    df["Distance_from_Previous_km"] = (
        df.groupby("Train_Number")["Cumulative_Distance_km"].diff().fillna(0).clip(lower=0)
    )
    return df


def get_rate(distance, first_station=False):
    if first_station or distance <= LOW_THRESHOLD:
        return 1.00
    if distance <= MEDIAN_THRESHOLD:
        return 0.90
    if distance <= UPPER_THRESHOLD:
        return 0.80
    return 0.70


def simulate_train(df, train_number, initial_delay):
    route = df[df["Train_Number"] == int(train_number)].sort_values("Cumulative_Distance_km").reset_index(drop=True)
    if route.empty:
        return pd.DataFrame()

    # A fixed seed per train makes results reproducible across runs and starting-delay scenarios.
    rng = random.Random(int(train_number))
    initial_component = float(initial_delay)
    new_component = 0.0
    previous_condition = None
    previous_average = None
    rows = []

    for i, station in route.iterrows():
        distance = float(station["Distance_from_Previous_km"])
        average = float(station["Average_Delay(min)"])
        rate = get_rate(distance, first_station=(i == 0))

        initial_component *= rate
        new_component *= rate

        weights = [max(0.0, float(station[c])) if pd.notna(station[c]) else 0.0 for c in CATEGORIES]
        if sum(weights) > 0:
            condition = rng.choices(DISPLAY_CATEGORIES, weights=weights, k=1)[0]
        else:
            condition = "Unknown"

        new_delay = 0.0
        current_rank = RANK.get(condition)
        previous_rank = RANK.get(previous_condition)
        if (
            current_rank is not None and previous_rank is not None
            and current_rank > previous_rank and previous_average is not None
        ):
            new_delay = max(0.0, average - previous_average)

        new_component += new_delay
        total = initial_component + new_component
        rows.append({
            "Train_Number": int(train_number),
            "Station_Name": station["Station_Name"],
            "Distance_from_Previous_km": round(distance, 2),
            "Carry_Forward_Rate": f"{rate:.0%}",
            "Historical_Station_Condition": condition,
            "Historical_Average_Delay": round(average, 2),
            "New_Delay": round(new_delay, 2),
            "Initial_Delay_Component": round(initial_component, 2),
            "New_Disruption_Component": round(new_component, 2),
            "Propagated_Delay": round(total, 2),
        })
        previous_condition = condition
        previous_average = average

    return pd.DataFrame(rows)


def run_all(df, starting_delay):
    all_results = []
    for train in sorted(df["Train_Number"].unique()):
        result = simulate_train(df, train, starting_delay)
        if result.empty:
            continue
        last = result.iloc[-1]
        all_results.append({
            "Train_Number": int(train),
            "Initial_Delay": float(starting_delay),
            "Initial_Component_at_End": float(last["Initial_Delay_Component"]),
            "New_Disruption_Component_at_End": float(last["New_Disruption_Component"]),
            "Final_Propagated_Delay": float(last["Propagated_Delay"]),
        })
    return pd.DataFrame(all_results)


def make_report(df, latest_station=None, latest_all=None, report_delay=None):
    trains_count = int(df["Train_Number"].nunique())
    category_means = df[CATEGORIES].mean(numeric_only=True)
    delay = df["Average_Delay(min)"]
    lines = [
        "# Train Delay Propagation Simulation Report",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Abstract",
        "This project analyzes historical station-level delay statistics for the matched train dataset and simulates how an initial delay and subsequent modeled disruptions propagate along each route. It uses distance-based carry-forward rates and historical station-condition percentages.",
        "",
        "## Dataset overview",
        f"- Matched trains: {trains_count}",
        f"- Station records: {len(df)}",
        f"- Mean historical station average delay: {delay.mean():.2f} minutes",
        f"- Median historical station average delay: {delay.median():.2f} minutes",
        f"- Minimum / maximum historical station average delay: {delay.min():.2f} / {delay.max():.2f} minutes",
        "",
        "### Mean station-condition percentages",
    ]
    for col in CATEGORIES:
        lines.append(f"- {col}: {category_means.get(col, float('nan')):.2f}%")
    lines += [
        "",
        "## Methodology",
        "1. Sort each train's stations by cumulative route distance.",
        "2. Calculate distance from the previous station by differencing cumulative distances. The first station is assigned 0 km and a 100% carry-forward rate.",
        "3. Apply carry-forward rates: distance up to 32 km = 100%; over 32 and up to 61 km = 90%; over 61 and up to 102.5 km = 80%; over 102.5 km = 70%.",
        "4. Sample a station condition using the historical category percentages as weights.",
        "5. Add a modeled new delay only when the sampled condition worsens compared with the previous station and the historical average delay increases. The amount is the positive difference between adjacent station historical average delays.",
        "6. Track the remaining initial-delay component and the new-disruption component separately. Their sum is the final propagated delay.",
        "",
        "## Assumptions and limitations",
        "- Carry-forward percentages are modeling assumptions, not estimated causal effects from a fitted statistical model.",
        "- Sampled station conditions and new delays are simulated; they are not observed incident-by-incident disruptions.",
        "- Historical average delay is used as a heuristic for the size of a modeled new disruption when the condition worsens.",
        "- The tool is a data-driven propagation simulation, not a validated real-world forecasting system. Its outputs should not be interpreted as guaranteed arrival-delay predictions.",
        "- The distance calculation depends on the correctness and ordering of cumulative-distance values in the input data.",
    ]

    if report_delay is not None:
        lines += ["", "## Latest simulation results", f"Starting delay: {float(report_delay):g} minutes"]
    if latest_all is not None and not latest_all.empty:
        sorted_all = latest_all.sort_values("Final_Propagated_Delay", ascending=False)
        lines += [
            f"- Trains simulated: {len(sorted_all)}",
            f"- Mean final propagated delay: {sorted_all['Final_Propagated_Delay'].mean():.2f} minutes",
            f"- Median final propagated delay: {sorted_all['Final_Propagated_Delay'].median():.2f} minutes",
            f"- Minimum / maximum final propagated delay: {sorted_all['Final_Propagated_Delay'].min():.2f} / {sorted_all['Final_Propagated_Delay'].max():.2f} minutes",
            "",
            "### Final delay by train",
            "| Train | Initial delay (min) | Initial component (min) | New-disruption component (min) | Final propagated delay (min) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for _, row in sorted_all.sort_values("Final_Propagated_Delay", ascending=False).iterrows():
            lines.append(f"| {int(row['Train_Number'])} | {row['Initial_Delay']:.2f} | {row['Initial_Component_at_End']:.2f} | {row['New_Disruption_Component_at_End']:.2f} | {row['Final_Propagated_Delay']:.2f} |")

    if latest_station is not None and not latest_station.empty:
        last = latest_station.iloc[-1]
        lines += [
            "",
            f"### Station-level result for train {int(last['Train_Number'])}",
            f"- Station records: {len(latest_station)}",
            f"- Final initial-delay component: {last['Initial_Delay_Component']:.2f} minutes",
            f"- Final new-disruption component: {last['New_Disruption_Component']:.2f} minutes",
            f"- Final propagated delay: {last['Propagated_Delay']:.2f} minutes",
        ]
    lines += ["", "## Reproducibility", "For a given train number, the random generator uses a fixed seed, so sampled station conditions remain reproducible for that train."]
    return "\n".join(lines)


def make_pdf(report_text):
    """Create a basic PDF report if reportlab is installed; otherwise return None."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    except ImportError:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.65*inch, bottomMargin=0.65*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallMono", parent=styles["BodyText"], fontName="Courier", fontSize=7, leading=9))
    story = []
    for paragraph in report_text.split("\n\n"):
        clean = paragraph.strip()
        if not clean:
            continue
        if clean.startswith("# "):
            story.append(Paragraph(clean[2:], styles["Title"]))
        elif clean.startswith("## "):
            story.append(Paragraph(clean[3:], styles["Heading2"]))
        elif clean.startswith("### "):
            story.append(Paragraph(clean[4:], styles["Heading3"]))
        elif clean.startswith("|---") or clean.startswith("| Train"):
            story.append(Preformatted(clean, styles["SmallMono"]))
        else:
            escaped = clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            escaped = escaped.replace("\n- ", "<br/>• ").replace("\n1. ", "<br/>1. ").replace("\n2. ", "<br/>2. ").replace("\n3. ", "<br/>3. ").replace("\n4. ", "<br/>4. ").replace("\n5. ", "<br/>5. ").replace("\n6. ", "<br/>6. ")
            story.append(Paragraph(escaped, styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


st.set_page_config(page_title="Train Delay Simulation", page_icon="🚆", layout="wide")
st.title("Train Delay Propagation Simulation")
st.caption("Historical analysis, interactive simulation, graphs, and report for the matched train dataset")

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info("Expected file: 01_Data/train_delay_23_trains.csv")
    st.stop()

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.info("Check the required columns in 01_Data/train_delay_23_trains.csv.")
    st.stop()

trains = sorted(df["Train_Number"].unique().tolist())
st.success(f"Loaded {len(trains)} trains and {len(df)} station records.")

tab_sim, tab_graphs, tab_report = st.tabs(["🚆 Simulation", "📊 Graphs & Visualization", "📄 Project Report"])

with tab_sim:
    mode = st.radio("Choose simulation mode", ["Single Train", "All Trains"], horizontal=True)
    st.info("Distance rules: up to 32 km = 100%; over 32 to 61 km = 90%; over 61 to 102.5 km = 80%; above 102.5 km = 70%.")

    if mode == "Single Train":
        col1, col2 = st.columns(2)
        with col1:
            default_index = trains.index(12346) if 12346 in trains else 0
            selected_train = st.selectbox("Select train number", trains, index=default_index)
        with col2:
            starting_delay = st.number_input("Initial delay (minutes)", min_value=0.0, max_value=1440.0, value=60.0, step=5.0, help="Enter any initial delay from 0 to 1440 minutes.")

        if st.button("Run Single Train Simulation", type="primary"):
            result = simulate_train(df, selected_train, starting_delay)
            st.session_state["single_result"] = result
            st.session_state["single_train"] = selected_train
            st.session_state["single_delay"] = starting_delay
            st.session_state["latest_station_result"] = result
            st.session_state["latest_run_delay"] = starting_delay
            st.session_state["latest_run_description"] = f"Single train {selected_train}"

        if "single_result" in st.session_state:
            result = st.session_state["single_result"]
            st.subheader(f"Train {st.session_state['single_train']} — {st.session_state['single_delay']:g} minute start")
            a, b, c = st.columns(3)
            a.metric("Station records", len(result))
            b.metric("Final propagated delay", f"{result['Propagated_Delay'].iloc[-1]:.2f} min")
            c.metric("Final initial component", f"{result['Initial_Delay_Component'].iloc[-1]:.2f} min")
            st.dataframe(result, use_container_width=True, hide_index=True, height=500)
            st.download_button("Download station results CSV", result.to_csv(index=False).encode("utf-8"), file_name=f"train_{st.session_state['single_train']}_results.csv", mime="text/csv")
    else:
        starting_delay = st.number_input("Initial delay for all trains (minutes)", min_value=0.0, max_value=1440.0, value=60.0, step=5.0, key="all_trains_initial_delay", help="Enter any starting delay from 0 to 1440 minutes.")
        if st.button("Run All Trains", type="primary"):
            with st.spinner("Running simulations for all matched trains..."):
                results = run_all(df, starting_delay)
                st.session_state["all_results"] = results
                st.session_state["all_results_delay"] = starting_delay
                st.session_state["latest_run_delay"] = starting_delay
                st.session_state["latest_run_description"] = "All trains"

        if "all_results" in st.session_state:
            results = st.session_state["all_results"]
            used_delay = st.session_state.get("all_results_delay", starting_delay)
            view = results.sort_values("Final_Propagated_Delay", ascending=False).round(2)
            st.subheader("Results for Every Train")
            st.caption(f"Initial delay used for this run: {used_delay:g} minutes")
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("Download all-train results CSV", results.to_csv(index=False).encode("utf-8"), file_name=f"all_train_results_{used_delay:g}min.csv", mime="text/csv")

            st.subheader("Inspect a Train Station by Station")
            inspect_col1, inspect_col2 = st.columns(2)
            with inspect_col1:
                train_to_inspect = st.selectbox("Train number", trains, key="inspect_train")
            with inspect_col2:
                inspect_delay = st.number_input("Starting delay for station details (minutes)", min_value=0.0, max_value=1440.0, value=float(used_delay), step=5.0, key="inspect_initial_delay")
            detail = simulate_train(df, train_to_inspect, inspect_delay)
            st.session_state["latest_station_result"] = detail
            st.session_state["latest_run_delay"] = inspect_delay
            st.dataframe(detail, use_container_width=True, hide_index=True, height=420)
            st.download_button("Download selected train station results", detail.to_csv(index=False).encode("utf-8"), file_name=f"train_{train_to_inspect}_stations_{inspect_delay:g}min.csv", mime="text/csv")

    with st.expander("Model assumptions"):
        st.markdown("""
        - Station distances are calculated from consecutive cumulative distances.
        - The first station uses a 100% carry-forward rate.
        - Historical conditions are sampled using the dataset's category percentages.
        - A new delay is added when the sampled condition worsens and historical average delay increases from the previous station.
        - Cancelled/Unknown is excluded from condition-rank comparisons.
        - These rules are simulation assumptions, not a validated real-world prediction model.
        """)

with tab_graphs:
    st.subheader("Historical Dataset Graphs")
    graph_left, graph_right = st.columns(2)
    with graph_left:
        st.markdown("**Distribution of historical average station delays**")
        hist = pd.cut(df["Average_Delay(min)"], bins=12).value_counts(sort=False)
        hist_df = hist.rename_axis("Delay range (minutes)").reset_index(name="Station records")
        hist_df["Delay range (minutes)"] = hist_df["Delay range (minutes)"].astype(str)
        st.bar_chart(hist_df.set_index("Delay range (minutes)"))
    with graph_right:
        st.markdown("**Mean historical station-condition percentages**")
        category_df = pd.DataFrame({"Condition": DISPLAY_CATEGORIES, "Mean percentage": [df[c].mean() for c in CATEGORIES]}).set_index("Condition")
        st.bar_chart(category_df)

    st.divider()
    st.subheader("Simulation Graphs — Based on the Latest Run")
    latest_station = st.session_state.get("latest_station_result")
    latest_all = st.session_state.get("all_results")
    if latest_station is not None and not latest_station.empty:
        train_num = int(latest_station["Train_Number"].iloc[0])
        st.markdown(f"**Train {train_num}: delay components by station**")
        component_chart = latest_station.set_index("Station_Name")[["Initial_Delay_Component", "New_Disruption_Component", "Propagated_Delay"]]
        st.line_chart(component_chart)
        st.caption("These lines use the current station-level simulation results. The total propagated delay equals the initial component plus the new-disruption component.")
    else:
        st.info("Run a Single Train simulation, or inspect a train in All Trains mode, to populate station-level simulation graphs.")

    if latest_all is not None and not latest_all.empty:
        st.markdown("**Final propagated delay by train (latest All Trains run)**")
        all_chart = latest_all.sort_values("Final_Propagated_Delay", ascending=False).copy()
        all_chart["Train"] = all_chart["Train_Number"].astype(str)
        st.bar_chart(all_chart.set_index("Train")[["Final_Propagated_Delay"]])
        st.caption(f"Starting delay for this all-train run: {st.session_state.get('all_results_delay', 0):g} minutes.")
    else:
        st.info("Run All Trains to generate the cross-train comparison graph.")

    st.download_button("Download historical delay data used for graphs", df.to_csv(index=False).encode("utf-8"), file_name="historical_analysis_data.csv", mime="text/csv")

with tab_report:
    st.subheader("Project Report")
    latest_station = st.session_state.get("latest_station_result")
    latest_all = st.session_state.get("all_results")
    report_delay = st.session_state.get("latest_run_delay")
    report_text = make_report(df, latest_station=latest_station, latest_all=latest_all, report_delay=report_delay)
    st.markdown("This report is generated from the dataset and the latest available simulation results. Run the simulation first if you want the latest run included.")
    st.markdown("**Report contents:** abstract, dataset overview, historical category statistics, methodology, distance rules, assumptions, limitations, and available simulation results.")
    with st.expander("Preview report"):
        st.markdown(report_text)
    st.download_button("Download report (Markdown)", report_text.encode("utf-8"), file_name="Train_Delay_Project_Report.md", mime="text/markdown")
    pdf_bytes = make_pdf(report_text)
    if pdf_bytes is not None:
        st.download_button("Download report (PDF)", pdf_bytes, file_name="Train_Delay_Project_Report.pdf", mime="application/pdf")
    else:
        st.warning("PDF generation needs ReportLab. Install it with: python -m pip install reportlab. You can download the Markdown report now.")

    st.subheader("Interpretation note")
    st.warning("This is a data-driven delay-propagation simulation. The carry-forward rates and new-disruption rule are assumptions; the output has not been validated as a real-world train-delay forecast.")
    