
from pathlib import Path
import random
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "01_Data" / "train_delay_23_trains.csv"

LOW_THRESHOLD = 32
MEDIAN_THRESHOLD = 61
UPPER_THRESHOLD = 102.5

SCENARIOS = [30, 60, 120]

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


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    df["Train_Number"] = pd.to_numeric(
        df["Train_Number"], errors="coerce"
    )
    df["Cumulative_Distance_km"] = pd.to_numeric(
        df["Cumulative_Distance_km"], errors="coerce"
    )
    df["Average_Delay(min)"] = pd.to_numeric(
        df["Average_Delay(min)"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Train_Number",
            "Cumulative_Distance_km",
            "Average_Delay(min)",
        ]
    ).copy()

    df["Train_Number"] = df["Train_Number"].astype(int)

    df = df.sort_values(
        ["Train_Number", "Cumulative_Distance_km"]
    ).reset_index(drop=True)

    df["Distance_from_Previous_km"] = (
        df.groupby("Train_Number")["Cumulative_Distance_km"]
        .diff()
        .fillna(0)
        .clip(lower=0)
    )

    return df


def get_rate(distance, first_station=False):
    if first_station or distance <= LOW_THRESHOLD:
        return 1.00
    elif distance <= MEDIAN_THRESHOLD:
        return 0.90
    elif distance <= UPPER_THRESHOLD:
        return 0.80
    return 0.70


def simulate_train(df, train_number, initial_delay):
    route = df[
        df["Train_Number"] == int(train_number)
    ].sort_values("Cumulative_Distance_km").reset_index(drop=True)

    if route.empty:
        return pd.DataFrame()

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

        weights = [
            max(0, float(station[c]))
            for c in CATEGORIES
        ]

        if sum(weights) > 0:
            condition = rng.choices(
                [
                    "Right Time",
                    "Slight Delay",
                    "Significant Delay",
                    "Cancelled/Unknown",
                ],
                weights=weights,
                k=1,
            )[0]
        else:
            condition = "Unknown"

        new_delay = 0.0
        current_rank = RANK.get(condition)
        previous_rank = RANK.get(previous_condition)

        if (
            current_rank is not None
            and previous_rank is not None
            and current_rank > previous_rank
            and previous_average is not None
        ):
            new_delay = max(0, average - previous_average)

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


def run_all(df):
    all_results = []

    for starting_delay in SCENARIOS:
        for train in sorted(df["Train_Number"].unique()):
            result = simulate_train(df, train, starting_delay)

            if result.empty:
                continue

            last = result.iloc[-1]

            all_results.append({
                "Train_Number": int(train),
                "Initial_Delay": starting_delay,
                "Initial_Component_at_End":
                    last["Initial_Delay_Component"],
                "New_Disruption_Component_at_End":
                    last["New_Disruption_Component"],
                "Final_Propagated_Delay":
                    last["Propagated_Delay"],
            })

    return pd.DataFrame(all_results)


st.set_page_config(
    page_title="Train Delay Simulation",
    page_icon="🚆",
    layout="wide",
)

st.title("Train Delay Propagation Simulation")
st.caption("Interactive simulation for the matched train dataset")

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.info(
        "Check that the CSV has Train_Number, Station_Name, "
        "Cumulative_Distance_km, Average_Delay(min), and all "
        "four delay-category percentage columns."
    )
    st.stop()

trains = sorted(df["Train_Number"].unique().tolist())

st.success(
    f"Loaded {len(trains)} trains and {len(df)} station records."
)

mode = st.radio(
    "Choose simulation mode",
    [
        "Single Train",
        "All Trains",
    ],
    horizontal=True,
)

st.info(
    "Distance rules: up to 32 km = 100%; "
    "up to 61 km = 90%; up to 102.5 km = 80%; "
    "above 102.5 km = 70%."
)

if mode == "Single Train":
    col1, col2 = st.columns(2)

    with col1:
        default_index = trains.index(12346) if 12346 in trains else 0
        selected_train = st.selectbox(
            "Select train number",
            trains,
            index=default_index,
        )

    with col2:
        starting_delay = st.number_input(
            "Initial delay (minutes)",
            min_value=0.0,
            max_value=1440.0,
            value=60.0,
            step=5.0,
            help="Enter any initial delay from 0 to 1440 minutes.",
        )

    if st.button("Run Single Train Simulation", type="primary"):
        result = simulate_train(
            df, selected_train, starting_delay
        )
        st.session_state["single_result"] = result
        st.session_state["single_train"] = selected_train
        st.session_state["single_delay"] = starting_delay

    if "single_result" in st.session_state:
        result = st.session_state["single_result"]

        st.subheader(
            f"Train {st.session_state['single_train']} — "
            f"{st.session_state['single_delay']} minute start"
        )

        a, b, c = st.columns(3)
        a.metric("Station records", len(result))
        b.metric(
            "Final propagated delay",
            f"{result['Propagated_Delay'].iloc[-1]:.2f} min",
        )
        c.metric(
            "Final initial component",
            f"{result['Initial_Delay_Component'].iloc[-1]:.2f} min",
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True,
            height=550,
        )

        st.download_button(
            "Download station results CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name=f"train_{st.session_state['single_train']}_results.csv",
            mime="text/csv",
        )


else:
    st.write(
        "Simulate all 23 matched trains using your chosen "
        "initial delay."
    )

    starting_delay = st.number_input(
        "Initial delay for all trains (minutes)",
        min_value=0.0,
        max_value=1440.0,
        value=60.0,
        step=5.0,
        key="all_trains_initial_delay",
        help="Enter any starting delay from 0 to 1440 minutes.",
    )


    
if st.button("Run All Trains", type="primary"):
    with st.spinner("Running simulations..."):
        all_results = []

        for train in trains:
            result = simulate_train(df, train, starting_delay)

            if not result.empty:
                last = result.iloc[-1]

                all_results.append({
                    "Train_Number": int(train),
                    "Initial_Delay": starting_delay,
                    "Initial_Component_at_End":
                        last["Initial_Delay_Component"],
                    "New_Disruption_Component_at_End":
                        last["New_Disruption_Component"],
                    "Final_Propagated_Delay":
                        last["Propagated_Delay"],
                })

        st.session_state["all_results"] = pd.DataFrame(all_results)

    if "all_results" in st.session_state:
        results = st.session_state["all_results"]

        
     
        view = results.sort_values(
    "Final_Propagated_Delay",
    ascending=False,
)

        st.dataframe(
            view, use_container_width=True, hide_index=True
        )

        st.download_button(
            "Download all-train results CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="all_train_simulation_results.csv",
            mime="text/csv",
        )

        st.subheader("Inspect a Train Station by Station")
        train_to_inspect = st.selectbox(
            "Train number",
            trains,
            key="inspect_train",
        )
        inspect_delay = starting_delay
        detail = simulate_train(
            df, train_to_inspect, inspect_delay
        )

        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
            height=450,
        )

        st.download_button(
            "Download selected train station results",
            detail.to_csv(index=False).encode("utf-8"),
            file_name=f"train_{train_to_inspect}_stations.csv",
            mime="text/csv",
        )

with st.expander("Model assumptions"):
    st.markdown("""
    - Station distances are calculated from consecutive cumulative distances.
    - The first station uses a 100% carry-forward rate.
    - Historical conditions are sampled using the dataset's category percentages.
    - New delay is added when the sampled condition worsens and historical
      average delay increases from the previous station.
    - Cancelled/Unknown is excluded from condition-rank comparisons.
    - These rules are simulation assumptions, not a validated real-world
      prediction model.
    """)
