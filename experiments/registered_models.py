"""Registry of every model in research/comparison_register.md (F1-F6, C1-C3).

Reads only the frozen validation records; nothing here selects or changes a
model. Scorers, the model-freeze manifest and the test analysis all use this
one list so that model identities, thresholds and hashes cannot drift.
"""
import json
from pathlib import Path

SEEDS = (42, 43, 44)
REPS = ("fft", "rgb", "dct", "wavelet", "residual", "dinov2")
VAL_CONDITIONS = ("ref", "jpeg_q80", "jpeg_q40")
TEST_QUALITIES = (95, 80, 60, 40, 20)
TEST_CONDITIONS = ("ref", *[f"jpeg_q{q}" for q in TEST_QUALITIES])
BOTH, DETECTION, ATTRIBUTION = ("detection", "attribution"), ("detection",), ("attribution",)

MAIN_THRESHOLDS = Path("experiments/results/main_validation_thresholds_seeds42_44.json")
CONTROL_THRESHOLDS = Path("experiments/results/control_validation_thresholds.json")
READOUT_REPORT = Path("experiments/runs/c23_control_source_readout_validation/report.json")
PROBE_REPORT = Path("experiments/results/c23_main_probe_compression_validation/report.json")
MATCHED_REPORT = "experiments/runs/c23_main_matched_q{}_probes_validation/report.json"
BORDER_REPORT = Path("experiments/runs/c23_control_border_probes_validation/report.json")
METADATA_REPORT = Path("experiments/runs/c23_control_metadata_probe_validation/report.json")


def _load(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    assert record.get("test_scores_used", False) is False, path
    return record


def rgb_models():
    """F1-F3 and the C1/C2 RGB controls: one ResNet-18 checkpoint each."""
    main, control = _load(MAIN_THRESHOLDS), _load(CONTROL_THRESHOLDS)
    models = []
    for seed in SEEDS:
        for family, mode in (("F1", "reference"), ("F2", "augmented"), ("F3", "paired_w1")):
            name = f"c23_main_{mode}_seed{seed}"
            entry = main["runs"][name]
            models.append(dict(id=f"{family}_s{seed}", family=family, seed=seed, kind="rgb",
                               run=Path("experiments/runs") / name, tasks=BOTH,
                               threshold=entry["threshold"], checkpoint_sha256=entry["checkpoint_sha256"],
                               validation_predictions_sha256=entry["validation_predictions_sha256"],
                               record=MAIN_THRESHOLDS.as_posix()))
        for family, control_name, tasks in (("C1-DET", "c1_det", DETECTION), ("C1-ATT", "c1_att", ATTRIBUTION),
                                            ("C2-BIN", "c2_bin", DETECTION)):
            name = f"c23_control_{control_name}_seed{seed}"
            entry = control["runs"][name]
            models.append(dict(id=f"{family}_s{seed}", family=family, seed=seed, kind="rgb",
                               run=Path("experiments/runs") / name, tasks=tasks,
                               threshold=entry.get("threshold"), checkpoint_sha256=entry["checkpoint_sha256"],
                               validation_predictions_sha256=entry["validation_predictions_sha256"],
                               record=CONTROL_THRESHOLDS.as_posix()))
    for model in models:
        model["validation_predictions"] = model["run"] / "validation_predictions.csv"
        model["validation_conditions"] = VAL_CONDITIONS
        assert (model["threshold"] is not None) == ("detection" in model["tasks"]), model["id"]
    return models


def readout_models():
    """Readout R on frozen F3 and C2-BIN encoders (attribution only; p_fake is the native head)."""
    report = _load(READOUT_REPORT)
    encoders = {m["id"]: m for m in rgb_models()}
    models = []
    for seed in SEEDS:
        for family, label, encoder in (("F3-R", f"f3_seed{seed}", f"F3_s{seed}"),
                                       ("C2-R", f"c2_bin_seed{seed}", f"C2-BIN_s{seed}")):
            entry, base = report["encoders"][label], encoders[encoder]
            assert entry["checkpoint_sha256"] == base["checkpoint_sha256"]
            folder = READOUT_REPORT.parent / label
            models.append(dict(id=f"{family}_s{seed}", family=family, seed=seed, kind="readout",
                               encoder=encoder, run=base["run"], checkpoint_sha256=base["checkpoint_sha256"],
                               parameters=folder / "readout_parameters.npz",
                               parameters_sha256=entry["parameters_sha256"], tasks=ATTRIBUTION,
                               threshold=None, validation_predictions=folder / "validation_predictions.csv",
                               validation_predictions_sha256=entry["validation_predictions_sha256"],
                               validation_conditions=VAL_CONDITIONS, record=READOUT_REPORT.as_posix()))
    return models


def probe_models():
    """F4 reference probes, F5/F6 known-quality probes and C3a face-masked probes."""
    models = []
    reference = _load(PROBE_REPORT)
    for rep in REPS:
        entry = reference["representations"][rep]
        models.append(dict(id=f"F4_{rep}", family="F4", rep=rep, kind="probe", masked=False,
                           parameters=Path(f"experiments/runs/c23_main_{rep}_probe_validation/probe_parameters.npz"),
                           parameters_sha256=entry["parameters_sha256"], threshold=entry["threshold"],
                           validation_predictions=PROBE_REPORT.parent / rep / "validation_predictions.csv",
                           validation_predictions_sha256=entry["validation_predictions_sha256"],
                           validation_conditions=VAL_CONDITIONS, record=PROBE_REPORT.as_posix()))
    for family, quality in (("F5", 80), ("F6", 40)):
        path = Path(MATCHED_REPORT.format(quality))
        report = _load(path)
        for rep in REPS:
            entry = report["representations"][rep]
            models.append(dict(id=f"{family}_{rep}", family=family, rep=rep, kind="probe", masked=False,
                               parameters=path.parent / rep / "probe_parameters.npz",
                               parameters_sha256=entry["parameters_sha256"], threshold=entry["threshold"],
                               validation_predictions=path.parent / rep / "validation_predictions.csv",
                               validation_predictions_sha256=entry["validation_predictions_sha256"],
                               validation_conditions=(f"jpeg_q{quality}",), record=path.as_posix()))
    border = _load(BORDER_REPORT)
    for rep in REPS:
        entry = border["representations"][rep]
        models.append(dict(id=f"C3a_{rep}", family="C3a", rep=rep, kind="probe", masked=True,
                           parameters=BORDER_REPORT.parent / rep / "probe_parameters.npz",
                           parameters_sha256=entry["parameters_sha256"], threshold=entry["threshold"],
                           validation_predictions=BORDER_REPORT.parent / rep / "validation_predictions.csv",
                           validation_predictions_sha256=entry["validation_predictions_sha256"],
                           validation_conditions=VAL_CONDITIONS, record=BORDER_REPORT.as_posix()))
    for model in models:
        model["tasks"] = BOTH
    return models


def metadata_model():
    report = _load(METADATA_REPORT)
    return dict(id="C3b", family="C3b", kind="metadata", tasks=BOTH,
                parameters=METADATA_REPORT.parent / "probe_parameters.npz",
                parameters_sha256=report["parameters_sha256"], threshold=report["threshold"],
                validation_predictions=METADATA_REPORT.parent / "validation_predictions.csv",
                validation_predictions_sha256=report["validation_predictions_sha256"],
                validation_conditions=("metadata",), record=METADATA_REPORT.as_posix())


def all_models():
    return [*rgb_models(), *readout_models(), *probe_models(), metadata_model()]


def records():
    return [MAIN_THRESHOLDS, CONTROL_THRESHOLDS, READOUT_REPORT, PROBE_REPORT,
            Path(MATCHED_REPORT.format(80)), Path(MATCHED_REPORT.format(40)), BORDER_REPORT, METADATA_REPORT]


if __name__ == "__main__":
    models = all_models()
    print(json.dumps({"models": len(models), "by_family": {
        f: sum(m["family"] == f for m in models) for f in dict.fromkeys(m["family"] for m in models)}}, indent=2))
