# What Survives Compression?

Research setup for comparing **deepfake detection** with **closed-set manipulation-method attribution** under additional lossy compression.

The initial experiment uses a FaceForensics++ C23 mirror, so its reference images already inherit H.264 compression. The study measures what remains useful after **additional** JPEG/video compression, rather than treating the starting media as uncompressed.

## Research documents

- [Current experiment specification](research/module1_specification.md)
- [Literature review and novelty assessment](research/literature_review.md)
- [Experimental protocol and hardware assessment](research/experimental_protocol.md)
- [Complete module roadmap](research/research_roadmap.md)
- [Module progress tracker](research/module_tracker.md)
- [Experiment matrix and run-record template](research/experiment_matrix.md)
- [Dataset audit and preparation results](research/dataset_audit_c23.md)
- [Corrected bibliography](research/references.bib)

## Current status

Preparation has established:

- An inventory of 5,000 primary videos across real, Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
- Official source-pair split mapping with no known input-ID conflicts.
- MP4 structure checks on all core videos and full decoding of fifteen sampled videos.
- A balanced 1,000-video pilot selection.
- An engineering test that extracted 800 reference crops from 100 videos, generated JPEG/recompression variants, and extracted FFT descriptors.

**No research classifier has been trained, and no scientific detection/attribution accuracy is reported.** The ten-group engineering test is not a statistically adequate final evaluation.

## Environment

Use CPython 3.12 in an isolated environment. The preprocessing lock records the versions actually used for the initial real-data preparation. From the repository root, with the intended CPython interpreter selected:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r experiments/environment-preprocessing.lock.txt
& .\.venv\Scripts\python.exe experiments/selftest.py
```

For training, install the additional packages in `experiments/requirements-research.txt` and a mutually compatible torch/torchvision GPU build. A documented reproducible starting build is torch 2.7.1/torchvision 0.22.1 with CUDA 12.8; this is not a claim that it is the latest release. Check current GPU/driver compatibility using the official PyTorch matrix and execute the included forward/backward benchmark before large runs.

```powershell
& .\.venv\Scripts\python.exe -m pip install -r experiments/requirements-research.txt
& .\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe experiments/benchmark_gpu.py --batch 8 --steps 100 --output experiments/results/gpu_batch8.json
```

Standalone FFmpeg/FFprobe with libx264/libx265 are required for the video re-encoding extension. PyAV already handles the preparation decoder. See [experiment instructions](experiments/README.md) for the full workflow and what remains unimplemented/unverified.

## Data and detector assets

Dataset media and generated outputs are excluded from Git. Obtain data through the documented sources and place it locally under `experiments/FaceForensics/FaceForensics++_C23` or supply another path to the scripts.

- [Official FaceForensics++ data, source definitions and access information](https://github.com/ondyari/FaceForensics)
- [C23 mirror used for the initial preparation](https://www.kaggle.com/datasets/xdxd003/ff-c23)
- [Frozen YuNet face-detector model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

The small official split files and their download provenance are included. Detector download provenance is included in `experiments/models/yunet_provenance.json`, while its ONNX weights are excluded. Use the pinned `face_detection_yunet_2023mar.onnx` checkpoint with the recorded hash for the initial preprocessing protocol.

Example audit after placing the data:

```powershell
& .\.venv\Scripts\python.exe experiments/audit_ffpp.py experiments/FaceForensics/FaceForensics++_C23 --official-splits experiments/data/splits --output experiments/data/audit_c23
```

Generated local manifests, crops, previews, feature caches, checkpoints, virtual environments, and dataset archives are excluded. Output tools generally refuse to overwrite a nonempty destination; use a fresh output directory when reproducing a check.

## Interpretation

The four attribution labels identify known facial manipulation pipelines within FF++. They do not identify individual generator checkpoints or unknown sources. Proposed paired learning and robustness comparisons remain experimental; a worldwide-first novelty claim has not been established.
