from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return "Flask is alive."

@app.route("/score-ilf-public", methods=["GET", "POST"])
def score_ilf_public():
    if request.method == "GET":
        return render_template("ilf_form.html")  # assumes your HTML form is named ilf_form.html

    amplitudes = request.form.getlist("amplitude[]", type=float)
    average_amplitude = sum(amplitudes) / len(amplitudes)
    change = amplitudes[-1] - amplitudes[0]
    percent_change = (change / amplitudes[0]) * 100 if amplitudes[0] != 0 else 0
    stability_index = round((max(amplitudes) - min(amplitudes)) / average_amplitude, 2)

    response_class = "Improved" if percent_change > 10 else "Flat"
    eeg_summary = "Stable pattern with mild gains."
    notes = "User may benefit from continued titration."
    shap_image = None

    # Plot
    plt.figure(figsize=(6, 3))
    plt.plot(amplitudes, marker='o', color='#0787bf')
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
        notes=notes,
        plot_url=plot_url,
        shap_image=shap_image
    )
