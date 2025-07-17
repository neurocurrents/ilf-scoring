from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def index():
    return "Flask is alive."

@app.route("/score-ilf-public", methods=["GET", "POST"])
def score_ilf_public():
    if request.method == "GET":
        return "<p>This endpoint is live. Submit using POST to receive scores.</p>"

    data = request.form
    scores = {
        "arousal": 0,
        "emotion": 0,
        "sleep": 0
    }

    for i in range(1, 4):
        scores["arousal"] += int(data.get(f"q{i}", 0))
    for i in range(4, 7):
        scores["emotion"] += int(data.get(f"q{i}", 0))
    for i in range(7, 10):
        scores["sleep"] += int(data.get(f"q{i}", 0))

    return {
        "message": "Scores received.",
        "scores": scores
    }

# If running locally (optional for debugging)
if __name__ == "__main__":
    app.run(debug=True)


