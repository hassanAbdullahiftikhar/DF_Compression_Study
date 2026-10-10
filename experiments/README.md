# Experimental starter kit

This is preparation for the protocol in `../research/experimental_protocol.md`. The downloaded C23 mirror is now present under `FaceForensics/FaceForensics++_C23`; its audit and real preprocessing checks are described in `../research/dataset_audit_c23.md`. There are no trained research classifiers or scientific accuracy results. PyTorch training and GPU speed remain unmeasured.

## Environment

Use a private virtual environment. The bundled Python identified during preparation is:

`C:/Users/hassa/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`

The following commands describe the full research-environment recipe. The current installation state is recorded immediately below; GPU-training packages are still pending. Run from `C:/Users/hassa/Desktop/Sem7/MBR` in PowerShell:

Update after dataset arrival: `.venv` now exists with the preprocessing packages listed in `environment-preprocessing.lock.txt`. The commands below are the remaining/full research-environment recipe, not evidence that torch is installed.

```powershell
& 'C:\Users\hassa\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r experiments/requirements-research.txt
& .\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe experiments/preflight.py
& .\.venv\Scripts\python.exe experiments/selftest.py
& .\.venv\Scripts\python.exe experiments/benchmark_gpu.py --batch 8 --steps 100 --output experiments/results/gpu_batch8.json
```

Torch 2.7.1/cu128 is a documented reproducible Blackwell-compatible baseline, not a recommendation that it is the latest release. A current mutually compatible torch/torchvision CUDA 13.x stack is another option; use the official matrix and save installed versions. Do not copy old research repositories' GPU binaries blindly. FFmpeg/FFprobe with libx264 and libx265 are needed for the video extension and are absent from the inspected PATH.

For just audit/JPEG/descriptors/selftest, the existing bundled NumPy/Pillow suffice:

```powershell
& 'C:\Users\hassa\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments/selftest.py
```

## Data preparation contract

Obtain FF++ access and released splits through its official repository. Keep its train/val/test JSON files in `experiments/data/splits`. Build a reference crop manifest based on `manifest_template.csv`: paths must be absolute; input images must be fixed 224×224 PNG crops; `condition=ref`; every manipulation's original IDs and all shared provenance must have the same split and group ID. Do not infer all provenance from only the first filename component.

For this C23 mirror, `audit_ffpp.py` now inventories the files and maps known inputs to official source-pair groups. `extract_reference_faces.py` performs deterministic frozen-YuNet extraction and logs reference geometry/failures; it was tested on 100 real videos. The full 1,000-video pilot's extraction is pending. `ref` means the C23-derived reference PNG; inherited compression history is recorded.

Completed data/audit paths and reproduction commands:

```powershell
& .\.venv\Scripts\python.exe experiments/audit_ffpp.py experiments/FaceForensics/FaceForensics++_C23 --official-splits experiments/data/splits --output experiments/data/audit_c23_new
& .\.venv\Scripts\python.exe experiments/extract_reference_faces.py experiments/data/audit_c23/videos_pilot.csv experiments/data/c23_pilot --model experiments/models/face_detection_yunet_2023mar.onnx --official-splits experiments/data/splits
```

The second command is the next full-pilot extraction. Do not reuse a nonempty output directory. `--groups-per-split 8 1 1` reproduces the smaller engineering check in a new directory. Public split lists can be obtained with `fetch_ffpp_splits.py`; saved copies and hashes already exist in `data/splits`.

```powershell
& .\.venv\Scripts\python.exe experiments/common.py experiments/data/reference.csv --official-splits experiments/data/splits
& .\.venv\Scripts\python.exe experiments/prepare_jpeg.py experiments/data/reference.csv experiments/data/jpeg_pilot --official-splits experiments/data/splits
& .\.venv\Scripts\python.exe experiments/features.py experiments/data/jpeg_pilot/manifest.csv experiments/features/fft_pilot --representation fft
```

`prepare_jpeg.py` creates reference plus five single-pass and three repeated-pass conditions, logs hashes/encoder settings, and refuses to overwrite existing outputs. It is for pilot/evaluation materialization; the trainer generates selected JPEG conditions on demand to conserve storage. Crop geometry is provided by the reference preprocessing; the JPEG script does not perform face detection.

`features.py` produces memory-mapped descriptors and a matching row manifest. Available families: `rgb`, `fft`, `dct`, `wavelet`, `residual`. They are low-cost deterministic probe baselines. RGB pooled pixels are not a learned ResNet embedding; residual is a declared Laplacian, not SRM; DCT statistics are recomputed from decoded crops. Frozen DINOv2 and stronger learned representation maps remain pending.

Once scikit-learn is installed, `probe_features.py` fits balanced logistic detection and source probes. It fits feature scaling only on training rows, chooses regularization on video-level validation scores, and saves test predictions and numeric model parameters without pickle. It was syntax-checked; fitting was not executed in the inspected dependency-light environment.

```powershell
& .\.venv\Scripts\python.exe experiments/probe_features.py experiments/features/fft_pilot experiments/runs/fft_reference --official-splits experiments/data/splits --train-conditions ref
& .\.venv\Scripts\python.exe experiments/probe_features.py experiments/features/fft_pilot experiments/runs/fft_mixed --official-splits experiments/data/splits --train-conditions ref jpeg_q80 jpeg_q40
```

By default, validation conditions match the training conditions. Keep held-out qualities out of model selection. The simple shared regularization selection is a starter protocol; predeclare any change to separate task-specific tuning.

## Paired RGB baseline

`model.py` and `train_paired.py` provide an ordinary ResNet-18 with binary, fake-source, and projection heads. Source loss excludes real samples. Both quality views receive supervised losses. Pair loss is cosine agreement of projections, and inverse-count loss weights address the 1-real/4-fake training imbalance. Validation uses only reference and training-domain qualities; test examples do not enter training or model selection.

```powershell
& .\.venv\Scripts\python.exe experiments/train_paired.py experiments/data/reference.csv experiments/runs/pair01 --official-splits experiments/data/splits --pair-weight 0.1 --pretrained
& .\.venv\Scripts\python.exe experiments/train_paired.py experiments/data/reference.csv experiments/runs/pair0 --official-splits experiments/data/splits --pair-weight 0 --pretrained
& .\.venv\Scripts\python.exe experiments/train_paired.py experiments/data/reference.csv experiments/runs/reference --official-splits experiments/data/splits --reference-only --pair-weight 0 --pretrained
& .\.venv\Scripts\python.exe experiments/predict_rgb.py experiments/data/reference.csv experiments/runs/pair01/best.pt experiments/results/pair01_test.csv --official-splits experiments/data/splits
& .\.venv\Scripts\python.exe experiments/metrics.py experiments/results/pair01_test.csv --threshold 0.5
```

The last threshold is an example. Choose the actual operating threshold on validation and freeze it before test. The starter's checkpoint ranking is average validation task balanced accuracy at 0.5; revise/predeclare selection rules before the final experiment. The `--pretrained` option downloads official ImageNet weights when absent. Training code was syntax-checked but could not be executed without its ML dependencies. Do not call its performance verified.

`metrics.py` averages probabilities by video and condition and calculates AUROC, binary balanced accuracy/macro F1, and fake-only source balanced accuracy/top-1/macro F1. Missing class support produces undefined balanced accuracy. EER, source confusion matrices, detection-gated attribution, paired lineage bootstrap, and final inferential plots remain analysis tasks.

## Video codec extension

```powershell
& .\.venv\Scripts\python.exe experiments/video_codec.py experiments/data/example_reference.mp4 experiments/data/transcoded/example_h264_crf23.mp4
```

The default prints a plan without encoding; add `--execute` when input data and FFmpeg are ready. It processes the complete video and records command, versions, and probe metadata. It refuses to overwrite a video. It does not perform face extraction or a distortion/PTS alignment audit. Preserve reference boxes/frame indices and validate timing before using the crops. Equal CRF between x264/x265 is not a matched-quality experiment.

## Verification status

- **Executed:** environment audit; official split mapping and structural checks on the 5,000 core files; full decoding of 15 sampled videos; reference face extraction on 100 videos; real JPEG/history generation on 800 crops; synthetic transform/metric/leakage checks; Python syntax checks.
- **Not executed:** full-pilot face extraction, PyTorch forward/backward or training, new video encoding, DINOv2/learned feature extraction, published baseline reproduction, and scientific statistical inference.
- **Next concrete gate:** finish full-pilot coverage and the isolated GPU research environment, then run the benchmark and training sanity check before the main grid.
