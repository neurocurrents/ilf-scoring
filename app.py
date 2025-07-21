
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Index route (sanity check)
@app.route("/")
def index():
    return "Flask is alive."

# GET route to show the ILF scoring form
@app.route("/score-ilf-public", methods=["GET"])
def show_ilf_form():
    return render_template("score_ilf.html")

# POST route to handle form submission and generate report
@app.route("/score-ilf-public", methods=["POST"])
def score_ilf_public():
    amplitudes = []

    # Handle uploaded ILF file if present
    ilf_file = request.files.get("ilf_file")
    if ilf_file and ilf_file.filename != "":
        try:
            df = pd.read_csv(ilf_file, header=None)
            amplitudes = df.iloc[:, 0].dropna().astype(float).tolist()
        except Exception as e:
            print("Error reading ILF file:", e)

    # Fallback to manual inputs
    if not amplitudes:
        for i in range(1, 41):
            val = request.form.get(f"amp{i}")
            if val:
                try:
                    amplitudes.append(float(val))
                except ValueError:
                    continue

    if not amplitudes:
        return "No valid amplitude values provided from upload or form.", 400

    # EEG Handling (CSV or summary)
    eeg_summary = request.form.get("eeg_data", "No EEG data provided.")
    eeg_file = request.files.get("eeg_file")
    if eeg_file and eeg_file.filename.lower().endswith(".csv"):
        try:
            eeg_df = pd.read_csv(eeg_file)
            eeg_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            eeg_stats = []
            for band in eeg_bands:
                cols = [col for col in eeg_df.columns if band in col.lower()]
                if cols:
                    mean_val = eeg_df[cols].mean().mean()
                    eeg_stats.append(f"{band.capitalize()} Mean: {mean_val:.3f}")
            if eeg_stats:
                eeg_summary = "\n".join(eeg_stats)
        except Exception as e:
            eeg_summary = f"EEG file error: {e}"

    # Behavioral Notes
    behavior_notes = request.form.get("behavior_notes", "No behavioral notes provided.")

    # Compute stats
    average_amplitude = sum(amplitudes) / len(amplitudes)
    change = amplitudes[-1] - amplitudes[0]
    percent_change = (change / amplitudes[0]) * 100 if amplitudes[0] != 0 else 0
    stability_index = round((max(amplitudes) - min(amplitudes)) / average_amplitude, 2)
    response_class = "Improved" if percent_change > 10 else "Flat"
    notes = "User may benefit from continued titration." if percent_change < 10 else "Maintain current protocol if stable."

    # Plotting
    plt.figure(figsize=(6, 3))
    plt.plot(amplitudes, marker='o', color='#007bff')
    plt.title("Amplitude Trend Over Sessions")
    plt.xlabel("Session")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plot_url = f"data:image/png;base64,{encoded}"

    return render_template(
        "report_ilf.html",
        average_amplitude=average_amplitude,
        percent_change=percent_change,
        stability_index=stability_index,
        response_class=response_class,
        eeg_summary=eeg_summary,
        behavior_notes=behavior_notes,
        notes=notes,
        plot_url=plot_url,
        shap_image=None
    )
