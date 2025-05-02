import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------
# ABOUT THE CHARTS:
#
# This script creates two types of visualizations from SIR simulation data:
#
# 1. SIR Line Plot:
#    - Shows the evolution of Susceptible (S), Infected (I), and Recovered (R)
#      over time as separate lines.
#    - This is a standard way to analyze compartmental epidemic models.
#    - It helps identify key behaviors like:
#        • When the infection peaks
#        • How quickly it spreads
#        • When the population reaches herd immunity or full recovery
#
# 2. Stacked Area Plot:
#    - Displays S, I, and R as stacked layers over time.
#    - The total area represents the entire population (always adds to 1).
#    - This chart is useful for visualizing how the **composition** of the population
#      shifts across states as the epidemic progresses.
#    - It's especially effective in presentations for showing change over time at a glance.
#
# Why these two?
# - The line plot gives clear, analytical insight.
# - The stacked area chart gives visual impact and a holistic view.
#
# These two together give both technical and visual clarity about how the
# system behaves across the simulation timeline.
# -------------------------------------------------------------------------------------

# ----------------------------
# Step 1: Load simulation results from CSV
# ----------------------------
def load_simulation_data(filename):
    """
    Load the CSV file containing SIR simulation results.
    Returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filename)
        # Ensure data is grouped by time step if the 'Process' column exists
        if 'Process' in df.columns:
            df = df.groupby("Time", as_index=False).mean()  # Aggregate by averaging

        # Clip values to ensure they are within valid bounds
        df["S"] = df["S"].clip(lower=0, upper=1)
        df["I"] = df["I"].clip(lower=0, upper=1)
        df["R"] = df["R"].clip(lower=0, upper=1)

        return df
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        return None

# ----------------------------
# Plot 1: Classic SIR Line Plot
# ----------------------------
def plot_sir_line(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Time"], df["S"], label="Susceptible (S)", linewidth=2)
    plt.plot(df["Time"], df["I"], label="Infected (I)", linewidth=2)
    plt.plot(df["Time"], df["R"], label="Recovered (R)", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Proportion of Population")
    plt.title("SIR Simulation Over Time")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sir_line_plot.png")  # Save plot to file
    print("Saved SIR line plot to sir_line_plot.png")


# ----------------------------
# Plot 2: Stacked Area Plot
# ----------------------------
def plot_sir_stacked_area(df):
    plt.figure(figsize=(10, 6))
    plt.stackplot(df["Time"], df["S"], df["I"], df["R"],
                  labels=["Susceptible", "Infected", "Recovered"],
                  alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Proportion of Population")
    plt.title("SIR Stacked Area Over Time")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("sir_stacked_area_plot.png")  # Save plot to file
    print("Saved SIR stacked area plot to sir_stacked_area_plot.png")


if __name__ == "__main__":
    # Load data
    filename = "./../simulation_results.csv"
    data = load_simulation_data(filename)

    if data is not None:
        # Generate plots
        plot_sir_line(data)
        plot_sir_stacked_area(data)
