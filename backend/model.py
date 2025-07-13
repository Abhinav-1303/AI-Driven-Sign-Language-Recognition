import torch
import numpy as np
from model_architecture import CTRGCN  # Import your custom model
import logging

# 🧾 Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔤 Map model output index to actual characters (A-Z, space, delete)
def index_to_char(index):
    if index == 26:  # Space character
        return " "
    elif index == 27:  # Delete (backspace) character
        return "⌫"
    else:
        return chr(index + 65)  # Converts index 0-25 to ASCII A-Z (65-90)

# 📦 Load the pre-trained CTR-GCN model
def load_model():
    logger.info("Loading CTR-GCN model...")
    model = CTRGCN(num_classes=28)  # 26 letters + space + delete (total 28 classes)

    try:
        # Load the model weights from file (on CPU for compatibility)
        model.load_state_dict(torch.load("best_ctr_gcn_model1.pth", map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode (important for dropout, batchnorm)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise  # Re-throw error after logging it

# 🤖 Predict sign from extracted hand keypoints
def predict_sign(model, keypoints):
    try:
        # 🧮 Convert input keypoints to NumPy array of type float32
        keypoints = np.array(keypoints, dtype=np.float32)

        # 🔁 Ensure correct shape: (21, 3) = 21 landmarks with (x, y, z)
        if keypoints.shape != (21, 3):
            logger.warning(f"Reshaping keypoints from {keypoints.shape} to (21, 3)")
            keypoints = keypoints.reshape(21, 3)

        # 📦 Flatten the array to 1D (shape: (63,)) and convert to tensor
        flattened = keypoints.reshape(-1)  # (21*3 = 63 values)
        input_tensor = torch.tensor(flattened, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 63)

        logger.info(f"Input tensor shape: {input_tensor.shape}")

        # 🔮 Make prediction with the model (no gradient needed)
        with torch.no_grad():
            output = model(input_tensor)  # Output shape: (1, 28)
            
            # 📈 Apply softmax to get probabilities for each class
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # 🔢 Get the index of the class with the highest probability
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # ✅ Only accept prediction if confidence is high enough
        if confidence > 0.4:  # This threshold is lowered from 0.7 for testing
            predicted_char = index_to_char(predicted_class)
            logger.info(f"Predicted: {predicted_char}, Confidence: {confidence:.4f}")
            return predicted_char, confidence
        else:
            logger.info(f"Low confidence prediction ({confidence:.4f}), ignoring")
            return None, confidence

    except Exception as e:
        # 🛑 Catch any unexpected errors during prediction
        logger.error(f"Error in prediction: {e}")
        return None, 0.0
