import torch
import numpy as np
import cv2
import mediapipe as mp
import torch.nn.functional as F
from model_architecture import CTRGCN
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load the trained model
model_path = r"E:\mainn\mainn\best_ctr_gcn_model1.pth"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found in {os.getcwd()}")
    exit(1)

# Load the model
model = CTRGCN(num_classes=28)  # 28 classes as in your training code
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print("Model loaded successfully")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not detected!")
    exit(1)

print("Webcam initialized successfully. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Process detected hands
        if results.multi_hand_landmarks:
            print("Hand detected!")
            for landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract keypoints
                keypoints = []
                for lm in landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z])
                
                # Convert to numpy array
                keypoints = np.array(keypoints, dtype=np.float32)
                print(f"Keypoints shape: {keypoints.shape}")  # Should be (21, 3)
                
                # Flatten keypoints for model input
                flattened = keypoints.flatten()
                print(f"Flattened shape: {flattened.shape}")  # Should be (63,)
                
                # Prepare input tensor
                input_tensor = torch.tensor(flattened, dtype=torch.float32).unsqueeze(0).to(device)
                print(f"Input tensor shape: {input_tensor.shape}")  # Should be (1, 63)
                
                try:
                    # Perform inference
                    with torch.no_grad():
                        print("Running model inference...")
                        outputs = model(input_tensor)
                        print(f"Raw model output: {outputs}")
                        
                        probabilities = F.softmax(outputs, dim=1)
                        print(f"Probabilities: {probabilities}")
                        
                        confidence, prediction = torch.max(probabilities, dim=1)
                        predicted_class = prediction.item()
                        confidence_value = confidence.item()
                        
                        print(f"Predicted class: {predicted_class}, Confidence: {confidence_value:.4f}")
                        
                        # Map to letter (A=0, B=1, etc.)
                        if predicted_class < 26:  # A-Z
                            predicted_letter = chr(predicted_class + 65)
                        elif predicted_class == 26:
                            predicted_letter = "SPACE"
                        elif predicted_class == 27:
                            predicted_letter = "DEL"
                        else:
                            predicted_letter = "?"
                        
                        print(f"Predicted letter: {predicted_letter}")
                        
                        # Display prediction on frame
                        cv2.putText(
                            frame, 
                            f"{predicted_letter} ({confidence_value:.2f})", 
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )
                
                except Exception as e:
                    print(f"Error during inference: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Display frame
        cv2.imshow("ASL Sign Detection", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application terminated")