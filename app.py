from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Index route (sanity check)
@app.route("/")
def index():
    return "Flask is alive."

# Route to show the ILF scoring form
@app.route("/score-ilf-public", methods=["GET"])
def show_ilf_form():
    return render_template("score_ilf.html")

# Route to handle form submission and generate report
@app.route("/score-ilf-public", methods=["POST"])
def score_ilf_public():
    # Collect amplitude data from form
    amplitudes = []
    for i in range(1, 41):
        val = request.form.get(f"amp{i}")
        if val:
            try:
                amplitudes.append(float(val))
            except ValueError:
                continue

    if not amplitudes:
        return "No valid amplitude values provided.", 400

    # Compute basic stats
    average_amplitude = sum(amplitudes) / len(amplitudes)
    change = amplitudes[-1] - amplitudes[0]
    percent_change = (change / amplitudes[0]) * 100 if amplitudes[0] != 0 else 0
    stability_index = round((max(amplitudes) - min(amplitudes)) / average_amplitude, 2)

    # Simple logic
    response_class = "Improved" if percent_change > 10 else "Flat"
    eeg_summary = request.form.get("eeg_data", "No EEG data provided.")
    behavior_notes = request.form.get("behavior_notes", "No behavioral notes provided.")
    notes = "User may benefit from continued titration." if percent_change < 10 else "Maintain current protocol if stable."

    # Generate trend plot
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

    # Render results
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
        shap_image=None  # reserved for future use
    )
