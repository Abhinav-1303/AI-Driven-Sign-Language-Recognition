import React, { useRef, useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const handsRef = useRef(null);

  const [screen, setScreen] = useState('landing');
  const [isPredicting, setIsPredicting] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState('');
  const [sentence, setSentence] = useState('');
  const [lastPrediction, setLastPrediction] = useState('');
  const [samePredictionCount, setSamePredictionCount] = useState(0);
  const [finalSentence, setFinalSentence] = useState('');
  const [showFinal, setShowFinal] = useState(false);

  const delayThreshold = 12;

  useEffect(() => {
    if (window.Hands && window.Camera && window.drawConnectors) {
      console.log("✅ MediaPipe scripts loaded");
    } else {
      console.log("❌ MediaPipe scripts not loaded");
    }
  }, []);

  const handleGetStarted = () => {
    setScreen('predict');
  };

  const handleStartPrediction = () => {
    if (!videoRef.current) return;
  
    console.log("📹 Starting webcam...");
    setIsPredicting(true);
  
    const hands = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
  
    // Adjust thresholds to match your offline code (minTrackingConfidence 0.5)
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5, 
    });
  
    hands.onResults(onResults);
    handsRef.current = hands;
  
    const camera = new window.Camera(videoRef.current, {
      onFrame: async () => {
        console.log("📤 Sending frame to MediaPipe...");
        if (handsRef.current && videoRef.current && isPredicting) {
          const image = videoRef.current;
          console.log("🎞 Sending image dimensions:", image.videoWidth, image.videoHeight);
          await handsRef.current.send({ image });
        }
      },
      width: 640,
      height: 480,
    });
  
    cameraRef.current = camera;
  
    camera.start().then(() => {
      console.log("📽 MediaPipe Camera started");
    }).catch((error) => {
      console.error("❌ Error starting camera:", error);
    });
  
    setTimeout(() => {
      if (videoRef.current) {
        console.log("📏 Video dimensions:", videoRef.current.videoWidth, videoRef.current.videoHeight);
      }
    }, 1000);
  
    setShowFinal(false);
    setFinalSentence('');
  };
  
  const handleStopPrediction = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
      console.log("⛔ Camera stopped");
    }
    setIsPredicting(false);
    processFinalSentence();
  };
  
  const handleReset = () => {
    setSentence('');
    setCurrentPrediction('');
    setLastPrediction('');
    setSamePredictionCount(0);
    setFinalSentence('');
    setShowFinal(false);
  };
  
  const updateSentence = (char) => {
    setSentence((prev) => {
      if (char === 'DEL') return prev.slice(0, -1);
      if (char === 'SPACE') return prev + ' ';
      if (char.length === 1) return prev + char;
      return prev;
    });
  };
  
  const speakSentence = () => {
    const utterance = new SpeechSynthesisUtterance(finalSentence || sentence);
    utterance.rate = 0.9;
    speechSynthesis.speak(utterance);
  };
  
  const processFinalSentence = () => {
    if (!sentence.trim()) return;
    const formatted = sentence
      .trim()
      .replace(/\s+/g, ' ')
      .replace(/(^\w|\.\s*\w)/g, (c) => c.toUpperCase());
  
    const final = formatted.endsWith('.') ? formatted : formatted + '.';
    setFinalSentence(final);
    setShowFinal(true);
  };
  
  const onResults = async (results) => {
    console.log("📡 onResults() called with results:", results);
  
    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext('2d');
  
    canvasElement.width = videoRef.current.videoWidth;
    canvasElement.height = videoRef.current.videoHeight;
  
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  
    if (!isPredicting || !results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      console.log("🙅 No hand detected");
      canvasCtx.restore();
      return;
    }
  
    console.log("🖐️ Hand landmarks detected!");
  
    for (const landmarks of results.multiHandLandmarks) {
      window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, { color: '#00FF00' });
      window.drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
    }
    canvasCtx.restore();
  
    // Flatten the landmarks to a keypoint array (63 elements)
    const keypoints = results.multiHandLandmarks[0].map((pt) => [pt.x, pt.y, pt.z]).flat();
  
    console.log("📍 Sending keypoints to backend:", keypoints);
  
    try {
      const res = await axios.post('http://localhost:5000/predict', { keypoints });
  
      if (res.data.error) {
        console.error("Error from backend:", res.data.error);
        return;
      }
  
      const predicted = res.data.prediction;
      const confidence = res.data.confidence;
  
      console.log("🤖 Prediction from server:", predicted, "| Confidence:", confidence);
  
      if (confidence < 0.8) return;
  
      setCurrentPrediction(predicted);
  
      if (predicted === lastPrediction) {
        setSamePredictionCount((count) => {
          if (count + 1 >= delayThreshold) {
            updateSentence(predicted);
            setLastPrediction('');
            return 0;
          }
          return count + 1;
        });
      } else {
        setLastPrediction(predicted);
        setSamePredictionCount(1);
      }
    } catch (err) {
      console.error('🔥 Prediction error:', err);
      setCurrentPrediction('Error');
    }
  };
  
  return screen === 'landing' ? (
    <div className="App landing">
      <h1>Signify: Where Sign Speaks</h1>
      <button className="get-started" onClick={handleGetStarted}>
        Get Started
      </button>
    </div>
  ) : (
    <div className="App">
      <h1>Signify: Where Sign Speaks</h1>
      <div className="video-container">
        <video ref={videoRef} autoPlay playsInline className="video" />
        <canvas ref={canvasRef} className="canvas" />
      </div>
      <div className="controls">
        <button className="start-button" onClick={handleStartPrediction} disabled={isPredicting}>
          Start Prediction
        </button>
        <button className="stop-button" onClick={handleStopPrediction} disabled={!isPredicting}>
          Stop Prediction
        </button>
        <button className="reset-button" onClick={handleReset}>
          Reset
        </button>
        {finalSentence && (
          <button className="start-button" onClick={speakSentence}>
            🔊 Speak
          </button>
        )}
      </div>
      <div className="output">
        <h2>Current Prediction:</h2>
        <p className="prediction">{currentPrediction}</p>
        <h3>Detected Sentence:</h3>
        <textarea value={sentence} readOnly className="text-area" rows={3} cols={60} />
        {showFinal && (
          <>
            <h3>Final Sentence:</h3>
            <p className="prediction">{finalSentence}</p>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
