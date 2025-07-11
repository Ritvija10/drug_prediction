from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model and encoders
with open("drug_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

le_sex = encoders["le_sex"]
le_bp = encoders["le_bp"]
le_cholesterol = encoders["le_cholesterol"]
le_drug = encoders["le_drug"]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        sex = le_sex.transform([request.form["sex"].upper()])[0]
        bp = le_bp.transform([request.form["bp"].upper()])[0]
        chol = le_cholesterol.transform([request.form["chol"].upper()])[0]
        na_to_k = float(request.form["na_to_k"])

        features = np.array([[age, sex, bp, chol, na_to_k]])
        prediction = clf.predict(features)
        drug = le_drug.inverse_transform(prediction)[0]

        return render_template("result.html", drug=drug)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
