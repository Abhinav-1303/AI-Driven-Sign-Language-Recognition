import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { Hands } from "@mediapipe/hands";
import * as cam from "@mediapipe/camera_utils";
import axios from "axios";

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const [text, setText] = useState("");
  const [confidence, setConfidence] = useState(null);

  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });

    hands.onResults(async (results) => {
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0].map(lm => [lm.x, lm.y, lm.z]);
        try {
          const response = await axios.post("http://localhost:5000/predict", {
            keypoints: landmarks
          });
          const { prediction, confidence } = response.data;
          if (prediction) {
            setText(prev => prev + prediction);
            setConfidence(confidence);
          }
        } catch (err) {
          console.error("Prediction error:", err);
        }
      }
    });

    if (webcamRef.current) {
      const camera = new cam.Camera(webcamRef.current.video, {
        onFrame: async () => await hands.send({ image: webcamRef.current.video }),
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, []);

  return (
    <div className="p-6">
      <Webcam ref={webcamRef} style={{ width: "640px", height: "480px" }} />
      <div className="mt-4">
        <h2 className="text-xl font-bold mb-2">Predicted Text:</h2>
        <textarea
          value={text}
          readOnly
          className="w-full p-3 border rounded shadow-md"
          rows={4}
        />
        {confidence && <p className="text-sm text-gray-600">Confidence: {confidence}</p>}
      </div>
    </div>
  );
};

export default WebcamCapture;
