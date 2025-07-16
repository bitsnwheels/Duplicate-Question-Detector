print("Starting Flask app...")

from flask import Flask, request, render_template
import pickle
import numpy as np

# Load models
with open('model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('w2v_model.pkl', 'rb') as f:
    w2v_model = pickle.load(f)

# Import your preprocessing logic
from utils import query_point_creator  # You will create this file next

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        q1 = request.form['question1']
        q2 = request.form['question2']
        try:
            query_features = query_point_creator(q1, q2, w2v_model)
            pred = rf.predict(query_features)[0]
            proba = rf.predict_proba(query_features)[0][1]  # Confidence for class 1

            prediction = f"Duplicate ({proba * 100:.1f}%)" if pred == 1 else f"Not Duplicate ({(1 - proba) * 100:.1f}%)"
        except Exception as e:
            prediction = f"Error: {e}"
        

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    import os

    port = os.environ.get("PORT")
    if port is None:
        print("PORT environment variable not found, defaulting to 5000")
        port = 5000
    else:
        port = int(port)
        print(f"Using PORT from environment: {port}")

    app.run(host="0.0.0.0", port=port)
