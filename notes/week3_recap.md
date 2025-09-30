# Week 3 Recap – First Training Insights

## Summary
- Completed a clean training loop on the corrected `my_first_dataset` split using `yolov8n.pt` as the backbone and logged the run at `runs/train/train_20250928_223128`.
- Formalised metric tracking in `reports/metrics.json` so each experiment captures its snapshot.
- Added a Streamlit dashboard to visualise training artifacts and review metrics in one place.

## Custom vs. Pretrained Baseline
- Custom model (`train_20250928_223128`): Precision 0.596, Recall 0.893, mAP50 0.888, mAP50-95 0.623.
- Pretrained yolov8n baseline on the same dataset still needs a fresh evaluation run (entries left as `null` in `reports/metrics.json`).
- Early impression: the finetuned model already matches mAP50 from the baseline recorded on coco8, but we need the direct baseline pass to confirm any recall gains on `my_first_dataset`.

## Notable Observations
- The model detects high-recall but slightly lower-precision, suggesting it is aggressive about flagging positives on the tiny dataset.
- Classes with few examples (e.g., umbrella, potted plant) still achieved near-perfect recall because coco8 annotations leak into the custom split; larger data will be needed for realistic behaviour.

## Challenges and Resolutions
- **Dataset configs**: Multiple YAML copies caused training to point at the wrong data. Solution: keep only `datasets/my_first_dataset.yaml` and timestamp every run’s config snapshot.
- **Package imports**: `imagedetection.py` initially failed when executed as a script; adding the project root to `sys.path` fixed the `detector` import.
- **Testing in CI**: OpenCV/Ultralytics imports crash under Rosetta, so the test suite now stubs those heavy modules before import and documents the limitation.

## Next Steps
1. Run `yolov8n.pt` in evaluation mode on `my_first_dataset` to populate the baseline row.
2. Collect FPS measurements at different image sizes and surface them in the Streamlit dashboard.
3. Expand the dataset beyond coco8 to validate the training pipeline at scale.
