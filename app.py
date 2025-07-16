# Neuro-AI EEG Learning Prediction Notebook
# Standalone version for scoring and SHAP interpretation

# ilf_score_route (Flask + ML)
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = "your-temp-key"  # Needed for session management

@app.route("/score-ilf-public", methods=["POST"])
def score_ilf_public():
    data = request.form
    scores = {"arousal": 0, "emotion": 0, "sleep": 0}

    for i in range(1, 4):
        scores["arousal"] += int(data.get(f"q{i}", 0))
    for i in range(4, 7):
        scores["emotion"] += int(data.get(f"q{i}", 0))
    for i in range(7, 10):
        scores["sleep"] += int(data.get(f"q{i}", 0))

    eeg_text = data.get("eeg_data")
    eeg_file = request.files.get("eeg_file")

    return {
        "status": "success",
        "scores": scores,
        "eeg_text_present": bool(eeg_text),
        "eeg_file_uploaded": eeg_file.filename if eeg_file else None,
    }

@app.route("/score-ilf", methods=["POST"])
def score_ilf():
    data = request.form
    scores = {"arousal": 0, "emotion": 0, "sleep": 0}

    for i in range(1, 4):
        scores["arousal"] += int(data.get(f"q{i}", 0))
    for i in range(4, 7):
        scores["emotion"] += int(data.get(f"q{i}", 0))
    for i in range(7, 10):
        scores["sleep"] += int(data.get(f"q{i}", 0))

    # Optional EEG text box and file upload
    eeg_text = data.get("eeg_data")
    eeg_file = request.files.get("eeg_file")

    print("Received EEG data:", eeg_text[:100] if eeg_text else "None")
    print("Received EEG file:", eeg_file.filename if eeg_file else "None")

return f"""
<html>
  <head><title>ILF Scoring Results</title></head>
  <body>
    <h2>Scoring Complete</h2>
    <p><strong>Arousal:</strong> {scores['arousal']}</p>
    <p><strong>Emotional Stability:</strong> {scores['emotion']}</p>
    <p><strong>Sleep & Recovery:</strong> {scores['sleep']}</p>
    <p><strong>EEG Text Provided:</strong> {'Yes' if eeg_text else 'No'}</p>
    <p><strong>EEG File Uploaded:</strong> {eeg_file.filename if eeg_file else 'No file'}</p>
  </body>
</html>
"""

@app.route("/score-ilf", methods=["POST"])
def score_ilf():
    data = request.form
    scores = {"arousal": 0, "emotion": 0, "sleep": 0}

    for i in range(1, 4):
        scores["arousal"] += int(data.get(f"responses[{i}]", 0))
    for i in range(4, 7):
        scores["emotion"] += int(data.get(f"responses[{i}]", 0))
    for i in range(7, 11):
        scores["sleep"] += int(data.get(f"responses[{i}]", 0))

    feedback = {
        "arousal": "High self-regulation" if scores["arousal"] >= 12 else "Needs support",
        "emotion": "Stable" if scores["emotion"] >= 12 else "Volatile",
        "sleep": "Restorative" if scores["sleep"] >= 12 else "Disrupted"
    }

    return render_template("report_ilf.html",
        arousal=scores["arousal"],
        emotion=scores["emotion"],
        sleep=scores["sleep"],
        feedback=feedback
    )


   # Flask route to handle ILF scoring + optional amplitude analysis
@app.route("/score-ilf", methods=["POST"])
def score_ilf():
    data = request.form

    # Parse domain ratings
    scores = {"arousal": 0, "emotion": 0, "sleep": 0}
    for i in range(1, 4):
        scores["arousal"] += int(data.get(f"responses[{i}]", 0))
    for i in range(4, 7):
        scores["emotion"] += int(data.get(f"responses[{i}]", 0))
    for i in range(7, 10):
        scores["sleep"] += int(data.get(f"responses[{i}]", 0))

    # Classify domain response
    def classify(score):
        if score >= 12:
            return "Strong Function"
        elif score >= 8:
            return "Moderate"
        else:
            return "Needs Support"

    ratings = {k: classify(v) for k, v in scores.items()}

    # Process optional amplitude input
    raw_amps = data.get("amplitudes", "").strip()
    amp_plot = None
    amp_category = None

    if raw_amps:
        try:
            amplitudes = [float(x) for x in raw_amps.split(",") if x.strip()]
            if 10 <= len(amplitudes) <= 40:
                from matplotlib import pyplot as plt
                import numpy as np
                sessions = list(range(1, len(amplitudes) + 1))
                slope = np.polyfit(sessions, amplitudes, 1)[0]
                volatility = np.std(amplitudes)
                
                if slope > 0.03 and volatility < 0.07:
                    amp_category = "Stable Improver"
                elif slope > 0.03 and volatility >= 0.07:
                    amp_category = "Volatile Improver"
                elif slope <= 0.03 and volatility < 0.07:
                    amp_category = "Flatline"
                else:
                    amp_category = "Unstable Trend"

                # Save plot
                plt.figure(figsize=(10, 4))
                plt.plot(sessions, amplitudes, marker="o", label="ILF Amplitude")
                plt.plot(sessions, np.poly1d(np.polyfit(sessions, amplitudes, 1))(sessions), linestyle="--", label="Trend")
                plt.title("ILF Amplitude Trend")
                plt.xlabel("Session")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.tight_layout()
                amp_plot = "ilf_amp_plot.png"
                plt.savefig(f"static/{amp_plot}")
                plt.close()
        except Exception as e:
            print("Amplitude parsing error:", e)

    return render_template("report_ilf.html",
                           arousal=ratings["arousal"],
                           emotion=ratings["emotion"],
                           sleep=ratings["sleep"],
                           amp_category=amp_category,
                           amp_plot=amp_plot)



@app.route("/tools")
def tool_selection():
    if not session.get('logged_in'):
        return redirect(url_for("login"))
    return render_template("select_tool.html")

# Simulate EEG dataset (same as your uploaded model)
np.random.seed(42)
n_subjects = 150
ilf_amp = np.random.normal(0.25, 0.08, n_subjects)
alpha_start = np.random.normal(35, 10, n_subjects)
smr_start = np.random.normal(5, 1.5, n_subjects)
theta_start = np.random.normal(8, 2, n_subjects)
th1 = np.random.uniform(20, 30, n_subjects)
th2 = np.random.uniform(10, 20, n_subjects)
th3 = np.random.uniform(5, 15, n_subjects)
sessions = np.random.randint(15, 45, n_subjects)

pc1 = (
    1.2 * alpha_start +
    1.5 * smr_start -
    1.8 * theta_start +
    20 * ilf_amp +
    0.3 * sessions +
    np.random.normal(0, 10, n_subjects)
)

sim_df = pd.DataFrame({
    "ILF_amp": ilf_amp,
    "Alpha_start": alpha_start,
    "SMR_start": smr_start,
    "Theta_start": theta_start,
    "Threshold1": th1,
    "Threshold2": th2,
    "Threshold3": th3,
    "n_sessions": sessions,
    "PC1_score": pc1
})

features = ["ILF_amp", "Alpha_start", "SMR_start", "Theta_start",
            "Threshold1", "Threshold2", "Threshold3", "n_sessions"]
X = sim_df[features]
y = sim_df["PC1_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_model)


def predict_learning_response(alpha, smr, theta, ilf, t1, t2, t3, sessions):
    input_df = pd.DataFrame([{
        "ILF_amp": ilf,
        "Alpha_start": alpha,
        "SMR_start": smr,
        "Theta_start": theta,
        "Threshold1": t1,
        "Threshold2": t2,
        "Threshold3": t3,
        "n_sessions": sessions
    }])
    return rf_model.predict(input_df)[0], explainer.shap_values(input_df)


@app.route("/score-ilf", methods=["POST"])
def score_ilf():
    data = request.form
    alpha = float(data.get("alpha", 35))
    smr = float(data.get("smr", 5))
    theta = float(data.get("theta", 8))
    ilf = float(data.get("ilf_amp", 0.25))
    t1 = float(data.get("t1", 24))
    t2 = float(data.get("t2", 14))
    t3 = float(data.get("t3", 10))
    sessions = int(data.get("n_sessions", 20))

    prediction, shap_vals = predict_learning_response(alpha, smr, theta, ilf, t1, t2, t3, sessions)

    # Generate a SHAP bar plot
    shap_df = pd.DataFrame([{
        "ILF_amp": ilf,
        "Alpha_start": alpha,
        "SMR_start": smr,
        "Theta_start": theta,
        "Threshold1": t1,
        "Threshold2": t2,
        "Threshold3": t3,
        "n_sessions": sessions
    }])

    shap.summary_plot(shap_vals, shap_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("static/ilf_shap.png")
    plt.close()

    return render_template("report_ilf.html", 
                           prediction=round(prediction, 2),
                           shap_image="ilf_shap.png")


@app.route("/ilf-form")
def ilf_form():
    return render_template("ilf_form.html")


