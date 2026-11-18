"""Single entry point for the project."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_Analysis"))
sys.path.insert(0, str(ROOT / "03_Simulation"))
from analysis import run_analysis
from simulation import run_simulation

def main():
    print("=" * 60)
    print("TRAIN DELAY PROPAGATION PROJECT")
    print("=" * 60)
    print("\n[1/2] Historical analysis")
    run_analysis()
    print("\n[2/2] Delay propagation simulation")
    run_simulation()
    print("\nComplete. Check 01_Data/ for CSV results and 04_Graphs/ for charts.")

if __name__ == "__main__":
    main()
