# Train Delay Prediction / Propagation Project

## Finalized dataset
The simulation uses **23 trains** matched to cumulative distance data. Original route CSVs for the wider 42-train dataset are retained in `01_Data/original_data/routes/`.

## Folders
- `01_Data/train_delay_23_trains.csv`: finalized distance-matched master dataset.
- `01_Data/original_data/`: original source data.
- `02_Analysis/analysis.py`: historical statistics and graphs.
- `03_Simulation/simulation.py`: V7 simulation and 30-, 60-, and 120-minute experiments.
- `04_Graphs/`: generated charts.
- `05_Report/report_outline.md`: report outline.
- `main.py`: runs analysis and simulation.

## Run in VS Code
1. Open this folder in VS Code.
2. In the terminal run `python -m pip install -r requirements.txt`.
3. Run `python main.py`.

## Model rules
- Segment distance up to 32 km: 100% carry-forward.
- Above 32 km and up to 61 km: 90%.
- Above 61 km and up to 102.5 km: 80%.
- Above 102.5 km: 70%.
- Station condition is sampled using the station's supplied category percentages.
- A modeled new disruption is added only when the sampled condition worsens and historical average delay increases between adjacent stations. The added amount is the positive difference between those historical averages.
- Original delay and new-disruption components are attenuated separately.

## Limitation
This is a **simulation model**, not a validated real-world prediction model. The carry-forward rates and disruption rule are assumptions. The supplied data do not include independent observed final-delay outcomes for validation, so no actual-versus-predicted validation is claimed.
