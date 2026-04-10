export interface HealthResponse {
  status: string;
  model_ready: boolean;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
}

export interface TrainResponse {
  message: string;
  samples: number;
  classes: number;
  train_accuracy: number;
  test_accuracy: number;
  total_matched_samples: number;
  skipped_non_letters: number;
  label_mode: string;
}
