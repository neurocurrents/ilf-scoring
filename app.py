from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return "Flask is alive."

@app.route("/score-ilf-public", methods=["POST"])
def score_ilf_public():
    # Collect amplitude values from named inputs amp1 to amp40
    amplitudes = []
    for i in range(1, 41):
        value = request.form.get(f"amp{i}")
        if value:
            try:
                amplitudes.append(float(value))
            except ValueError:
                pass  # ignore non-numeric values

    # Optional fields
    eeg_data = request.form.get("eeg_data", "")
    behavior_notes = request.form.get("behavior_notes", "")

    # Basic calculations
    average_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else 0
    change = amplitudes[-1] - amplitudes[0] if len(amplitudes) >= 2 else 0
    percent_change = (change / amplitudes[0]) * 100 if len(amplitudes) >= 2 and amplitudes[0] != 0 else 0
    stability_index = round((max(amplitudes) - min(amplitudes)) / average_amplitude, 2) if average_amplitude else 0
    response_class = "Improved" if percent_change > 10 else "Flat"
    eeg_summary = "Stable pattern with mild gains."
    notes = "User may benefit from continued titration."
    shap_image = None  # Placeholder for SHAP logic

    # Create the trend plot
    import io
    import matplotlib.pyplot as plt
    import base64
    buf = io.BytesIO()
    plt.figure(figsize=(6, 3))
    plt.plot(amplitudes, marker='o', color='#007bff')
    plt.title("Amplitude Trend Over Sessions")
    plt.xlabel("Session")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plot_url = f"data:image/png;base64,{encoded}"

    # Render result
    return render_template("report_ilf.html",
        average_amplitude=average_amplitude,
        percent_change=percent_change,
        stability_index=stability_index,
        response_class=response_class,
        eeg_summary=eeg_summary,
        notes=notes,
        plot_url=plot_url,
        shap_image=shap_image,
        eeg=eeg_data,
        behavior=behavior_notes
    )
