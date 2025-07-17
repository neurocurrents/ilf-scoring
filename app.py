from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("score_ilf.html")

@app.route("/score-ilf-public", methods=["POST"])
def score_ilf():
    amplitudes = []
    eeg_data = request.form.get("eeg_data", "")
    behavior_notes = request.form.get("behavior_notes", "")

    for i in range(1, 41):
        value = request.form.get(f"amp{i}", "")
        if value:
            try:
                amplitudes.append(float(value))
            except ValueError:
                amplitudes.append(None)
        else:
            amplitudes.append(None)

    # Compute summary metrics
    valid_values = [v for v in amplitudes if v is not None]
    if valid_values:
        avg_amp = round(np.mean(valid_values), 2)
        trend = "increasing" if valid_values[-1] > valid_values[0] else "decreasing"
        change = round(valid_values[-1] - valid_values[0], 2)
    else:
        avg_amp = 0
        trend = "no data"
        change = 0

    return render_template("score_ilf.html", result={
        "average": avg_amp,
        "trend": trend,
        "change": change,
        "behavior": behavior_notes,
        "eeg": eeg_data
    })

if __name__ == "__main__":
    app.run(debug=True)


