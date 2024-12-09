from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Tải mô hình đã lưu từ MLflow
model = mlflow.sklearn.load_model("fraud_detection_model_mlflow")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)