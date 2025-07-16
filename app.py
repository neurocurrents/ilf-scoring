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


