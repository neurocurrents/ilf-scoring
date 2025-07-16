# Neuro-AI EEG Learning Prediction Notebook
# Standalone version for scoring and SHAP interpretation

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = "your-temp-key"  # Needed for session management

# Route for public ILF scoring form (with GET and POST)
@app.route("/score-ilf-public", methods=["GET", "POST"])
def score_ilf_public():
    if request.method == "GET":
        return "<p>This endpoint is live. Submit using POST to receive scores.</p>"

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

    print("Received EEG data:", eeg_text[:100] if eeg_text else "None")
    print("Received EEG file:", eeg_file.filename if eeg_file else "None")

    return f"""
    <html>
      <head>
        <title>ILF Scoring Results</title>
        <style>
          body {{ font-family: Arial; padding: 40px; background: #f9f9f9; }}
          .box {{ background: #fff; border-radius: 12px; padding: 30px; max-width: 600px; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
          h2 {{ text-align: center; color: #003366; }}
          p {{ font-size: 18px; margin-bottom: 10px; }}
          strong {{ color: #222; }}
        </style>
      </head>
      <body>
        <div class="box">
          <h2>ILF Scoring Summary</h2>
          <p><strong>Arousal Regulation:</strong> {scores['arousal']}</p>
          <p><strong>Emotional Stability:</strong> {scores['emotion']}</p>
          <p><strong>Sleep & Recovery:</strong> {scores['sleep']}</p>
          <hr>
          <p><strong>EEG Text Provided:</strong> {'Yes' if eeg_text else 'No'}</p>
          <p><strong>EEG File Uploaded:</strong> {eeg_file.filename if eeg_file else 'No file'}</p>
        </div>
      </body>
    </html>
    """

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

    shap_df = pd.DataFrame([{ "ILF_amp": ilf, "Alpha_start": alpha, "SMR_start": smr, "Theta_start": theta,
                              "Threshold1": t1, "Threshold2": t2, "Threshold3": t3, "n_sessions": sessions }])
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

@app.route("/tools")
def tool_selection():
    if not session.get('logged_in'):
        return redirect(url_for("login"))
    return render_template("select_tool.html")

# Simulated EEG dataset
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
    input_df = pd.DataFrame([{ "ILF_amp": ilf, "Alpha_start": alpha, "SMR_start": smr,
                                "Theta_start": theta, "Threshold1": t1, "Threshold2": t2,
                                "Threshold3": t3, "n_sessions": sessions }])
    return rf_model.predict(input_df)[0], explainer.shap_values(input_df)
