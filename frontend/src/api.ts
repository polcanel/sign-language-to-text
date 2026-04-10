import type { HealthResponse, PredictionResponse, TrainResponse } from "./types";

async function parseJson<T>(response: Response): Promise<T> {
  const data = (await response.json()) as T & { detail?: string };
  if (!response.ok) {
    const message = (data as { detail?: string }).detail || "Request failed";
    throw new Error(message);
  }
  return data;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch("/health");
  return parseJson<HealthResponse>(response);
}

export async function trainModel(): Promise<TrainResponse> {
  const response = await fetch("/api/train", { method: "POST" });
  return parseJson<TrainResponse>(response);
}

export async function predictVideo(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/predict-video", {
    method: "POST",
    body: formData
  });
  return parseJson<PredictionResponse>(response);
}
