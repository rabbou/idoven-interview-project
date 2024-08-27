import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.data_preprocessing import load_raw_data


def load_and_plot_ecg(index: int, metadata: pd.DataFrame, path: str, sampling_rate: int = 100) -> None:
    """
    Load and plot all 12 leads of an ECG signal in a single combined graph with a red grid.

    Parameters:
    - index (int): The index of the ECG entry in the metadata DataFrame.
    - metadata (pd.DataFrame): The metadata DataFrame containing ECG information, including file paths and annotations.
    - sampling_rate (int): The sampling rate of the ECG data (100 or 500 Hz). Defaults to 100 Hz.
    - path (str): The base path where ECG files are stored.

    Returns:
    - None: The function displays a plot of the 12 ECG leads but does not return any values.
    """
    ecg_data = load_raw_data(metadata.iloc[[index]], sampling_rate, path)
    ecg_signals = ecg_data[0]

    patient_id = metadata.iloc[index]["patient_id"]
    ecg_id = metadata.index[index]

    annotations = metadata.iloc[index]["scp_codes"]
    print(f"Annotations (SCP Codes) for Patient ID {patient_id}, ECG ID {ecg_id}:", annotations)

    plt.figure(figsize=(8, 11))
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    y_offsets = np.linspace(0, -40, 12)  # Vertical offsets to separate each ECG lead

    # Plot each of the 12 leads
    for i in range(12):
        plt.plot(ecg_signals[:, i] + y_offsets[i], color="black", linewidth=0.7)
        plt.text(-5, y_offsets[i], lead_names[i], fontsize=9, ha="right", va="center", color="black")

    # Configure the grid and axes for the ECG plot
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(sampling_rate))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(sampling_rate / 5))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1.0))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    plt.grid(True, which="major", color="red", linestyle="-", linewidth=0.3)
    plt.grid(True, which="minor", color="red", linestyle="-", linewidth=0.1)

    # Set up x-axis labels as time in seconds
    x_ticks = np.arange(0, ecg_signals.shape[0] + 1, sampling_rate)
    x_labels = x_ticks / sampling_rate
    plt.xticks(x_ticks, labels=[f"{label:.1f}" for label in x_labels])

    # Remove y-axis numerical labels
    plt.gca().yaxis.set_major_formatter(plt.NullFormatter())

    plt.title(f"ECG Signal - 12 Leads Combined (Patient ID: {patient_id}, ECG ID: {ecg_id})", fontsize=12)
    plt.xlabel("Time (seconds)", fontsize=10)

    plt.show()
