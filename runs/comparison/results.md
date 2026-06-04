# Model Comparison

Dataset: `configs/lisa.yaml`  |  Split: `val`  |  imgsz: 640

## Overall Metrics

| Model | mAP50 | mAP50-95 | Precision | Recall | Params(M) | Inference(ms) |
| --- | --- | --- | --- | --- | --- | --- |
| yolov8:phase2_smoke_full_best | 0.0007 | 0.0006 | 0.0077 | 0.0310 | 3.0 | 53.1 |

## Per-Class AP@50 (coarse: stop / yield / warning / speed-limit)

| Class | yolov8:phase2_smoke_full_best |
| --- | --- |
| stop | 0.0000 |
| yield | 0.0000 |
| warning | 0.0017 |
| speed-limit | 0.0003 |
