import "./App.scss";
import { useEffect, useMemo, useRef, useState } from "react";
import { getHealth, predictVideo, trainModel } from "./api";
import type { HealthResponse, PredictionResponse } from "./types";

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    void refreshHealth();
    return () => stopCamera();
  }, []);

  const statusLabel = useMemo(() => {
    if (!health) return "connecting";
    return health.model_ready ? "model ready" : "model not trained";
  }, [health]);

  async function refreshHealth(): Promise<void> {
    try {
      const data = await getHealth();
      setHealth(data);
    } catch {
      setError("Backend is not available. Run python main.py first.");
    }
  }

  async function handleTrain(): Promise<void> {
    setIsTraining(true);
    setError("");
    setResult(null);

    try {
      const data = await trainModel();
      setResult({
        prediction: `✓ Model trained: ${data.samples} samples, ${data.classes} classes`,
        confidence: data.test_accuracy
      });
      await refreshHealth();
    } catch (trainError) {
      setError((trainError as Error).message);
    } finally {
      setIsTraining(false);
    }
  }

  async function handlePredictUpload(): Promise<void> {
    if (!videoFile) {
      setError("Choose a video file first.");
      return;
    }
    await sendForPrediction(videoFile);
  }

  async function handlePredictRecording(): Promise<void> {
    if (!recordingBlob) {
      setError("Record a clip first.");
      return;
    }
    const file = new File([recordingBlob], "recording.webm", {
      type: recordingBlob.type || "video/webm"
    });
    await sendForPrediction(file);
  }

  async function sendForPrediction(file: File): Promise<void> {
    setIsPredicting(true);
    setError("");
    setResult(null);

    try {
      const data = await predictVideo(file);
      setResult(data);
    } catch (predictError) {
      setError((predictError as Error).message);
    } finally {
      setIsPredicting(false);
    }
  }

  async function startCamera(): Promise<void> {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (cameraError) {
      setError(`Cannot access camera: ${(cameraError as Error).message}`);
    }
  }

  function stopCamera(): void {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function startRecording(): void {
    if (!streamRef.current) {
      setError("Start camera first.");
      return;
    }

    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(streamRef.current, { mimeType: "video/webm" });

    recorder.ondataavailable = (event: BlobEvent) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    recorder.onstop = () => {
      setRecordingBlob(new Blob(chunks, { type: "video/webm" }));
    };

    recorderRef.current = recorder;
    recorder.start();
    setIsRecording(true);
    setError("");
  }

  function stopRecording(): void {
    if (recorderRef.current && isRecording) {
      recorderRef.current.stop();
      setIsRecording(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>Sign Language to Text</h1>
        <p>
          Train the model on slovo_keypoints dataset, then upload or record a clip to get text prediction.
        </p>
        <div className="status">Backend: {health?.status || "..."} | {statusLabel}</div>
        <button disabled={isTraining || isPredicting} onClick={() => void handleTrain()}>
          {isTraining ? "Training model..." : "Train model (letters only)"}
        </button>
      </header>

      <section className="grid">
        <article className="card">
          <h2>Upload Video</h2>
          <p>Pick a sign video and send it to backend for inference.</p>
          <input
            type="file"
            accept="video/*"
            onChange={(event) => setVideoFile(event.target.files?.[0] || null)}
          />
          <button disabled={isTraining || isPredicting} onClick={() => void handlePredictUpload()}>
            {isPredicting ? "Processing..." : "Predict uploaded video"}
          </button>
        </article>

        <article className="card">
          <h2>Record Camera</h2>
          <p>Record 3 to 6 seconds of one gesture, then run prediction.</p>
          <div className="controls">
            <button onClick={() => void startCamera()} disabled={isRecording}>Start camera</button>
            <button onClick={stopCamera} disabled={isRecording}>Stop camera</button>
            {!isRecording ? (
              <button onClick={startRecording}> 🔴 Start recording</button>
            ) : (
              <button onClick={stopRecording}>🟥 Stop recording</button>
            )}
          </div>
          <video className="preview" ref={videoRef} autoPlay playsInline muted />
          <button
            disabled={isTraining || isPredicting || !recordingBlob}
            onClick={() => void handlePredictRecording()}
          >
            Predict recording
          </button>
        </article>
      </section>

      {result && (
        <section className="result">
          <div className="resultTitle">Result</div>
          <div>{result.prediction}</div>
          <div>
            {result.prediction.startsWith("✓ Model trained") ? "Test accuracy" : "Confidence"}: {(result.confidence * 100).toFixed(1)}%
          </div>
        </section>
      )}

      {error && <section className="error">{error}</section>}
    </div>
  );
}
