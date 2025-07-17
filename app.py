from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("score_ilf.html")

@app.route("/score-ilf-public", methods=["POST"])
def score_ilf_public():
    amplitude_values = [request.form.get(f"amp{i}") for i in range(1, 41)]
    amplitudes = [float(v) for v in amplitude_values if v not in (None, "")]

    if amplitudes:
        import numpy as np
        average_amplitude = round(np.mean(amplitudes), 2)
        trend = (
            "increasing" if amplitudes[-1] > amplitudes[0]
            else "decreasing" if amplitudes[-1] < amplitudes[0]
            else "stable"
        )

        change = round(amplitudes[-1] - amplitudes[0], 2)
        percent_change = round(((amplitudes[-1] - amplitudes[0]) / amplitudes[0]) * 100, 1)

        # Stability Index: correlation with ideal ramp
        ideal_ramp = np.linspace(amplitudes[0], amplitudes[-1], len(amplitudes))
        correlation = np.corrcoef(amplitudes, ideal_ramp)[0, 1] if len(amplitudes) > 2 else 0
        stability_index = round(correlation, 2)

        # Classification
        if abs(change) > 0.2 and stability_index >= 0.6:
            response_class = "Strong Learner"
        elif abs(change) > 0.1:
            response_class = "Moderate Response"
        else:
            response_class = "Minimal/Unclear Learning"

    else:
        average_amplitude = change = percent_change = stability_index = response_class = trend = None

    return render_template("report_ilf.html",
        average_amplitude=average_amplitude,
        trend=trend,
        change=change,
        percent_change=percent_change,
        stability_index=stability_index,
        response_class=response_class
    )


if __name__ == "__main__":
    app.run(debug=True)


