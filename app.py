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
    amplitudes = request.form.getlist("amplitude[]", type=float)
    average_amplitude = sum(amplitudes) / len(amplitudes)
    change = amplitudes[-1] - amplitudes[0]
    percent_change = (change / amplitudes[0]) * 100 if amplitudes[0] != 0 else 0
    stability_index = round((max(amplitudes) - min(amplitudes)) / average_amplitude, 2)

    # Placeholder logic for additional variables
    response_class = "Improved" if percent_change > 10 else "Flat"
    eeg_summary = "Stable pattern with mild gains."
    notes = "User may benefit from continued titration."
    shap_image = None  # Placeholder for SHAP image or logic

    # Generate trend plot
    plt.figure(figsize=(6, 3))
    plt.plot(amplitudes, marker='o', color='#007bff')
    plt.title("Amplitude Trend Over Sessions")
    plt.xlabel("Session")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # Save plot to a base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    trend_plot = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return render_template("report_ilf.html",
                           average_amplitude=average_amplitude,
                           trend=amplitudes,
                           change=change,
                           percent_change=percent_change,
                           stability_index=stability_index,
                           response_class=response_class,
                           eeg_summary=eeg_summary,
                           notes=notes,
                           shap_image=shap_image,
                           trend_plot=trend_plot)

if __name__ == "__main__":
    app.run(debug=True)



