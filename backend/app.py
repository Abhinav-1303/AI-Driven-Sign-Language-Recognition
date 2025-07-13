from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from model_architecture import CTRGCN
from model import predict_sign

app = Flask(__name__)
CORS(app)

MODEL_PATH = r"E:\mainn\mainn\backend\best_ctr_gcn_model1.pth"
model = CTRGCN(num_classes=28)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

@app.route("/", methods=["GET"])
def home():
    return "Flask backend is running. Use /predict to post keypoints."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🌍 /predict endpoint called")
        data = request.get_json()
        keypoints = data.get("keypoints", None)
        
        if not keypoints:
            print("❌ Error: Missing keypoints in request")
            return jsonify({"error": "Missing keypoints"}), 400
        
        print("✅ Received keypoints:", keypoints)
        
        keypoints = np.array(keypoints, dtype=np.float32)
        if keypoints.shape == (21, 3):
            keypoints = keypoints.flatten()
        elif keypoints.shape != (63,):
            print(f"❌ Error: Invalid keypoint shape received: {keypoints.shape}")
            return jsonify({"error": f"Invalid keypoint shape: {keypoints.shape}"}), 400

        predicted_char, confidence = predict_sign(model, keypoints)
        print(f"🤖 Predicted: {predicted_char} with confidence: {confidence:.2f}")
        return jsonify({
            "prediction": predicted_char,
            "confidence": confidence
        })
    except Exception as e:
        print("🔥 Error processing request:", str(e))
        return jsonify({"error": str(e)}), 500

def predict_sign(model, keypoints):
    # Convert the flattened keypoints (1,63) to a tensor
    keypoints = torch.tensor(keypoints).float().unsqueeze(0)
    output = model(keypoints)
    _, predicted_class = torch.max(output, 1)
    confidence = torch.softmax(output, dim=1)[0][predicted_class].item()
    return predicted_class.item(), confidence

if __name__ == "__main__":
    app.run(debug=True)
