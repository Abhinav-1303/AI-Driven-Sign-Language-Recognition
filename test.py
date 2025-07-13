import requests
import numpy as np

keypoints = np.random.rand(21, 3).tolist()  # Dummy data simulating 21 keypoints

response = requests.post("http://localhost:5000/predict", json={"keypoints": keypoints})

print("Status:", response.status_code)
print("Response:", response.json())
