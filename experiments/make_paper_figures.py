"""Paper figures and tables from the stored aggregate results.

Reads only research/main_results_test.json, main_results_extensions.json and main_results_dfdm.json,
so it needs no raw predictions, crops or models. The recorded environment has no plotting package,
so figures are written as plain SVG; --render converts each to PDF (vector) and PNG with a headless
Edge or Chrome. Outputs go to paper/figures and paper/tables; existing files are kept unless --force.

    .\\DeepFake\\Scripts\\python.exe experiments/make_paper_figures.py --render
"""
import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
SEEDS = (42, 43, 44)
JPEG = ("ref", "jpeg_q95", "jpeg_q80", "jpeg_q60", "jpeg_q40", "jpeg_q20")
JPEG_LABELS = ("Ref", "Q95", "Q80", "Q60", "Q40", "Q20")
DF_COLS = JPEG + ("h264_crf10", "h264_crf23")
DF_LABELS = JPEG_LABELS + ("CRF 10", "CRF 23")
CODECS = ("h264_crf18", "h264_crf23", "h264_crf32", "h264_crf40", "h265_crf23", "h265_crf32")
CODEC_LABELS = ("H.264 18", "H.264 23", "H.264 32", "H.264 40", "H.265 23", "H.265 32")

# Reference palette (dataviz skill); the three categorical slots validate all-pairs for colour-vision deficiency.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ("#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
FONT = "Arial, Helvetica, sans-serif"


def load(name):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


TEST = load("main_results_test.json")["analysis"]
EXT_FILE = load("main_results_extensions.json")
EXT, CLIP, JEPA = EXT_FILE["analysis"], EXT_FILE["X7_clip"]["analysis"], EXT_FILE["X8_jepa"]["analysis"]
DF_FILE = load("main_results_dfdm.json")
DFDM, E10 = DF_FILE["dfdm_analysis"], DF_FILE["e10_analysis"]


def retention(value, ref, chance):
    """Chance-adjusted retention; undefined when the reference margin is below 0.05 (register rule)."""
    if value is None or ref is None or ref - chance < 0.05:
        return None
    return (value - chance) / (ref - chance)


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------------------------- SVG


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width, height):
        self.width, self.height, self.parts = width, height, []

    def line(self, x1, y1, x2, y2, color=AXIS, width=0.6):
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                          f'stroke="{color}" stroke-width="{width}"/>')

    def path(self, points, color, width=1.4, dash=None):
        segments, current = [], []
        for p in points + [None]:
            if p is None:
                if len(current) > 1:
                    segments.append(current)
                current = []
            else:
                current.append(p)
        for seg in segments:
            d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in seg)
            self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                              f'stroke-linejoin="round" stroke-linecap="round"'
                              + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")

    def rect(self, x, y, w, h, fill, stroke="none", rx=0):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="0.6"/>')

    def text(self, x, y, s, size=7.5, anchor="start", color=INK, weight="normal", rotate=None, italic=False, raw=False):
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        style = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
                          f'font-weight="{weight}"{style}{transform}>{s if raw else esc(s)}</text>')

    def marker(self, shape, x, y, color, r=2.8, opacity=1.0):
        ring = f'stroke="#ffffff" stroke-width="0.8" fill="{color}"' + (f' fill-opacity="{opacity}"' if opacity < 1 else "")
        if shape == "circle":
            self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" {ring}/>')
        elif shape == "square":
            s = r * 1.75
            self.parts.append(f'<rect x="{x - s / 2:.2f}" y="{y - s / 2:.2f}" width="{s:.2f}" height="{s:.2f}" {ring}/>')
        elif shape == "diamond":
            s = r * 1.3
            self.parts.append(f'<path d="M{x:.2f},{y - s:.2f} L{x + s:.2f},{y:.2f} L{x:.2f},{y + s:.2f} '
                              f'L{x - s:.2f},{y:.2f} Z" {ring}/>')

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}pt" height="{self.height}pt" '
                f'viewBox="0 0 {self.width} {self.height}" font-family="{FONT}">\n'
                f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>\n'
                + "\n".join(self.parts) + "\n</svg>\n")


def sub(text, size=5.6):
    """Subscript markup for Svg.text(raw=True)."""
    return f'<tspan baseline-shift="-1.8" font-size="{size}">{esc(text)}</tspan>'


class Axes:
    """A plot area with a linear y scale and categorical x positions (floats allowed for gaps)."""

    def __init__(self, svg, x, y, w, h, span, ylim, pad=10):
        self.svg, self.x, self.y, self.w, self.h, self.span, self.pad = svg, x, y, w, h, span, pad
        self.lo, self.hi = ylim

    def X(self, i):
        return self.x + self.pad + i * (self.w - 2 * self.pad) / self.span

    def Y(self, v):
        return self.y + self.h - (v - self.lo) / (self.hi - self.lo) * self.h

    def frame(self, yticks, xticks, yfmt="{:g}", title=None, ylabel=None, tag=None, ytick_labels=True):
        svg = self.svg
        for t in yticks:
            svg.line(self.x, self.Y(t), self.x + self.w, self.Y(t), GRID, 0.5)
            if ytick_labels:
                svg.text(self.x - 3, self.Y(t) + 2.5, yfmt.format(t).replace("-", "−"), 6.8, "end", MUTED)
        svg.line(self.x, self.y + self.h, self.x + self.w, self.y + self.h, AXIS, 0.7)
        for i, label in xticks:
            svg.text(self.X(i), self.y + self.h + 9, label, 6.8, "middle", INK2)
        if title:
            svg.text(self.x, self.y - 6, title, 8, "start", INK, "bold")
        if tag:
            svg.text(self.x + self.w, self.y - 6, tag, 8, "end", INK2, "bold")
        if ylabel:
            svg.text(self.x - 27, self.y + self.h / 2, ylabel, 7.2, "middle", INK2, rotate=-90, raw="<tspan" in ylabel)

    def hline(self, v, label=None, color=AXIS, left=False):
        self.svg.line(self.x, self.Y(v), self.x + self.w, self.Y(v), color, 0.8)
        if label:
            x, anchor = (self.x + 2, "start") if left else (self.x + self.w - 2, "end")
            self.svg.text(x, self.Y(v) - 2.5, label, 6.2, anchor, MUTED, italic=True)

    def series(self, ys, color, dash=None, shape="circle", xs=None, offset=0.0):
        xs = xs if xs is not None else range(len(ys))
        pts = [None if v is None else (self.X(i) + offset, self.Y(v)) for i, v in zip(xs, ys)]
        self.svg.path(pts, color, 1.5, dash)
        for p in pts:
            if p is not None:
                self.svg.marker(shape, p[0], p[1], color)

    def whiskers(self, lows, highs, color, xs=None, offset=0.0):
        xs = xs if xs is not None else range(len(lows))
        for i, lo, hi in zip(xs, lows, highs):
            if lo is not None and hi is not None and hi - lo > 1e-9:
                self.svg.line(self.X(i) + offset, self.Y(lo), self.X(i) + offset, self.Y(hi), color, 0.9)


def legend(svg, x, y, items, size=7.2, gap=14):
    """items: (label, color, dash, shape). Draws a horizontal legend and returns its end x."""
    for label, color, dash, shape in items:
        if dash is not False:
            svg.path([(x, y - 2.5), (x + 16, y - 2.5)], color, 1.5, dash)
        if shape:
            svg.marker(shape, x + 8, y - 2.5, color)
        svg.text(x + 20, y, label, size, "start", INK2)
        x += 20 + len(label) * size * 0.5 + gap
    return x


def ticks(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + k * step, 6) for k in range(n + 1)]


def nice_limits(values, step, floor_hi=1.0, floor_lo=0.0):
    values = [v for v in values if v is not None]
    lo = min(floor_lo, math.floor(min(values) / step) * step)
    hi = max(floor_hi, math.ceil(max(values) / step) * step)
    return lo, hi


# ------------------------------------------------------------------------------------------ figures


def ffpp_ba_retention(family, metric):
    """Per-seed BA retention for the FF++ test curves (ResNet-18 from the registered record, EB4 from X6)."""
    out = []
    for s in SEEDS:
        if family == "F1":
            points = {c: TEST["model_intervals"][f"F1_s{s}"][c][f"{metric}_ba"]["point"] for c in JPEG}
        else:
            points = {c: EXT["X6_efficientnet_b4"]["curves"][f"{family}_s{s}"][c][f"{metric}_ba"]["point"] for c in JPEG}
        chance = 0.5 if metric == "detection" else 0.25
        out.append([retention(points[c], points["ref"], chance) for c in JPEG])
    return out


def ffpp_auc_retention(family, metric):
    return [[EXT["X1a_threshold_free"][f"{family}_s{s}"][c][f"{metric}_auc_retention"]["point"] for c in JPEG]
            for s in SEEDS]


def dfdm_ba_retention(family, metric):
    return [[DFDM["curves"][f"{family}_s{s}"][c][f"{metric}_retention"]["point"] for c in JPEG] for s in SEEDS]


def dfdm_auc_retention(family, metric):
    out = []
    for s in SEEDS:
        curve = DFDM["curves"][f"{family}_s{s}"]
        ref = curve["ref"][f"{metric}_auc"]["point"]
        out.append([retention(curve[c][f"{metric}_auc"]["point"], ref, 0.5) for c in JPEG])
    return out


def check_records():
    """Recomputed values must match the stored registered summaries before anything is drawn."""
    summary = TEST["exploratory"]["seed_summary"]["F1"]
    for metric in ("detection", "attribution"):
        mine = ffpp_ba_retention("F1", metric)
        for k, c in enumerate(JPEG):
            stored = summary[c][f"{metric}_retention"]["values_by_seed"]
            assert np.allclose([m[k] for m in mine], stored), (metric, c)
    for s in SEEDS:
        stored = TEST["primary_contrast"]["P1_RGB"]["per_seed"][f"F1_s{s}"]["point"]
        det, att = ffpp_ba_retention("F1", "detection")[SEEDS.index(s)], ffpp_ba_retention("F1", "attribution")[SEEDS.index(s)]
        assert np.isclose(att[-1] - det[-1], stored)
        stored = EXT["X6_efficientnet_b4"]["P1_rule_at_q20"]["per_seed"][f"EB4-F1_s{s}"]["point"]
        det, att = ffpp_ba_retention("EB4-F1", "detection")[SEEDS.index(s)], ffpp_ba_retention("EB4-F1", "attribution")[SEEDS.index(s)]
        assert np.isclose(att[-1] - det[-1], stored)
        for fam in ("D-F1", "D-EB4-F1"):
            det, att = dfdm_auc_retention(fam, "detection")[SEEDS.index(s)], dfdm_auc_retention(fam, "attribution")[SEEDS.index(s)]
            assert np.isclose(att[-1] - det[-1], DFDM["D_S1_delta_auc_q20"][f"{fam}_s{s}"]["point"])
            stored = DFDM["D_P1"][fam]["per_seed"][f"{fam}_s{s}"]["point"]
            det, att = dfdm_ba_retention(fam, "detection")[SEEDS.index(s)], dfdm_ba_retention(fam, "attribution")[SEEDS.index(s)]
            assert np.isclose(att[-1] - det[-1], stored)


def retention_panel(ax, groups):
    """groups: (seed_curves, colour, dash, shape, offset). Mean line plus a seed min-max whisker."""
    for curves, color, dash, shape, offset in groups:
        cols = list(zip(*curves))
        ax.whiskers([min((v for v in c if v is not None), default=None) for c in cols],
                    [max((v for v in c if v is not None), default=None) for c in cols], color, offset=offset)
    for curves, color, dash, shape, offset in groups:
        ax.series([mean(c) for c in zip(*curves)], color, dash, shape, offset=offset)


def fig_retention():
    svg = Svg(504, 384)
    panels = {
        ("ffpp", "ba"): (ffpp_ba_retention, ("F1", "EB4-F1")),
        ("ffpp", "auc"): (ffpp_auc_retention, ("F1", "EB4-F1")),
        ("dfdm", "ba"): (dfdm_ba_retention, ("D-F1", "D-EB4-F1")),
        ("dfdm", "auc"): (dfdm_auc_retention, ("D-F1", "D-EB4-F1")),
    }
    data = {key: {(fam, m): fn(fam, m) for fam in fams for m in ("detection", "attribution")}
            for key, (fn, fams) in panels.items()}
    every = [v for d in data.values() for curves in d.values() for row in curves for v in row]
    ylim = nice_limits(every, 0.25, 1.0, 0.0)
    legend(svg, 52, 14, [("Detection", BLUE, None, None), ("Attribution", ORANGE, None, None),
                         ("ResNet-18", INK2, None, "circle"), ("EfficientNet-B4", INK2, "3.5,2", "square")])
    svg.text(500, 14, "line = seed mean; whisker = seed range", 6.6, "end", MUTED, italic=True)
    columns = {"ffpp": (52, "FF++: four methods"), "dfdm": (298, "DFDM: five generators")}
    rows = {"ba": (52, "Retention S (BA, frozen threshold)"),
            "auc": (220, f"Retention S{sub('AUC')} (threshold-free)")}
    tags = iter("abcd")
    for row, (y, ylabel) in rows.items():
        for col, (x, title) in columns.items():
            ax = Axes(svg, x, y, 196, 118, len(JPEG) - 1, ylim)
            ax.frame(ticks(*ylim, 0.25), list(enumerate(JPEG_LABELS)), title=title, tag=f"({next(tags)})",
                     ylabel=ylabel if col == "ffpp" else None)
            ax.hline(0, "chance", left=True)
            ax.hline(1, "no loss")
            fams = panels[(col, row)][1]
            retention_panel(ax, [
                (data[(col, row)][(fams[0], "detection")], BLUE, None, "circle", -3.6),
                (data[(col, row)][(fams[1], "detection")], BLUE, "3.5,2", "square", -1.2),
                (data[(col, row)][(fams[0], "attribution")], ORANGE, None, "circle", 1.2),
                (data[(col, row)][(fams[1], "attribution")], ORANGE, "3.5,2", "square", 3.6),
            ])
    svg.text(52 + (298 + 196 - 52) / 2, 378, "Additional JPEG quality (reference-trained models)", 7.4, "middle", INK2)
    return svg


def fig_granularity():
    svg = Svg(252, 194)
    summary = TEST["exploratory"]["seed_summary"]["F1"]
    ff_keys = (("Det.", "detection_retention"), ("Family", "family_retention"), ("DF/FS", "within_swap_retention"),
               ("4-way", "attribution_retention"), ("F2F/NT", "within_reenactment_retention"))
    cats = []
    for i, (label, key) in enumerate(ff_keys):
        cats.append((i, label, {c: summary[c][key]["values_by_seed"] for c in ("jpeg_q60", "jpeg_q20")}))
    for i, (label, key) in zip((5.7, 6.7), (("Det.", "detection_retention"), ("5-way", "attribution_retention"))):
        cats.append((i, label, {c: [DFDM["curves"][f"D-F1_s{s}"][c][key]["point"] for s in SEEDS]
                                for c in ("jpeg_q60", "jpeg_q20")}))
    values = [v for _, _, d in cats for vs in d.values() for v in vs]
    ylim = nice_limits(values, 0.25, 1.0, 0.0)
    ax = Axes(svg, 40, 34, 204, 128, 6.7, ylim, pad=12)
    ax.frame(ticks(*ylim, 0.25), [(i, label) for i, label, _ in cats], ylabel="Retention S (0 chance, 1 no loss)")
    ax.hline(0)
    ax.hline(1)
    legend(svg, 40, 14, [("Q60", BLUE, False, "circle"), ("Q20", ORANGE, False, "square")])
    svg.text(244, 14, "marker = seed mean; dots = seeds", 6.4, "end", MUTED, italic=True)
    sep = (ax.X(4) + ax.X(5.7)) / 2
    svg.line(sep, 34, sep, 162 + 22, GRID, 0.7)
    for c, color, shape, off in (("jpeg_q60", BLUE, "circle", -4.5), ("jpeg_q20", ORANGE, "square", 4.5)):
        for i, _, d in cats:
            for v in d[c]:
                svg.parts.append(f'<circle cx="{ax.X(i) + off:.2f}" cy="{ax.Y(v):.2f}" r="1.3" fill="{color}" fill-opacity="0.45"/>')
            svg.marker(shape, ax.X(i) + off, ax.Y(mean(d[c])), color, 3.0)
    svg.text((ax.X(0) + ax.X(4)) / 2, 186, "FF++ (ResNet-18, F1)", 7.2, "middle", INK, "bold")
    svg.text((ax.X(5.7) + ax.X(6.7)) / 2, 186, "DFDM (D-F1)", 7.2, "middle", INK, "bold")
    return svg


def fig_confusion():
    svg = Svg(504, 170)
    ff_names = ("DF", "F2F", "FS", "NT")
    df_names = ("FSW", "LW", "IAE", "DFK", "H128")
    assert tuple(DFDM["D_S5_confusion"]["generators"]) == ("faceswap", "lightweight", "iae", "dfaker", "dfl_h128")

    def ffpp(c):
        return np.sum([TEST["exploratory"]["point_metrics"][f"F1_s{s}"][c]["source_confusion"] for s in SEEDS], axis=0)

    def dfdm(c):
        return np.sum([DFDM["D_S5_confusion"]["matrices"][f"D-F1_s{s}_{c}"] for s in SEEDS], axis=0)

    panels = ((34, "(a) FF++, Ref", ffpp("ref"), ff_names, True), (156, "(b) FF++, Q20", ffpp("jpeg_q20"), ff_names, False),
              (290, "(c) DFDM, Ref", dfdm("ref"), df_names, True), (406, "(d) DFDM, Q20", dfdm("jpeg_q20"), df_names, False))
    for x, title, m, names, row_labels in panels:
        m = np.asarray(m, dtype=float)
        assert len(set(m.sum(axis=1))) == 1, "rows are the true classes, equal per class"
        p = m / m.sum(axis=1, keepdims=True)
        cell = 92 / len(names)
        y0 = 44
        svg.text(x, 12, title, 8, "start", INK, "bold")
        svg.text(x + 46, 24, "predicted", 6.4, "middle", MUTED, italic=True)
        for j, n in enumerate(names):
            svg.text(x + (j + 0.5) * cell, y0 - 4, n, 6.4, "middle", INK2)
            if row_labels:
                svg.text(x - 3, y0 + (j + 0.5) * cell + 2.3, n, 6.4, "end", INK2)
        if row_labels:
            svg.text(x - 26 if len(names) == 5 else x - 22, y0 + 46, "true", 6.4, "middle", MUTED, rotate=-90, italic=True)
        for i in range(len(names)):
            for j in range(len(names)):
                v = p[i, j]
                fill = RAMP[min(len(RAMP) - 1, int(round(v * (len(RAMP) - 1))))]
                svg.rect(x + j * cell + 0.5, y0 + i * cell + 0.5, cell - 1, cell - 1, fill)
                svg.text(x + (j + 0.5) * cell, y0 + (i + 0.5) * cell + 2.4, f"{100 * v:.0f}", 6.4, "middle",
                         "#ffffff" if v >= 0.5 else INK)
    svg.text(252, 156, "Row-normalised percentages, reference-trained ResNet-18 (F1 / D-F1), three seeds pooled.",
             6.8, "middle", MUTED, italic=True)
    return svg


def fig_delta_auc():
    rows = [("FF++ (four methods)", None)]
    x1a = EXT["X1a_threshold_free"]
    for fam, label in (("F1", "ResNet-18"), ("EB4-F1", "EfficientNet-B4")):
        for s in SEEDS:
            rows.append((f"{label}, seed {s}", (x1a[f"{fam}_s{s}"]["jpeg_q20"]["delta_auc"], BLUE)))
    rows.append(("CLIP ViT-L/14 probe", (CLIP["X7_REF_delta_auc"]["jpeg_q20"]["delta_auc"], BLUE)))
    rows.append(("I-JEPA ViT-H/14 probe", (JEPA["X8_REF_delta_auc"]["jpeg_q20"]["delta_auc"], BLUE)))
    for rep, label in (("dinov2", "DINOv2 probe"), ("rgb", "Pooled-RGB probe"), ("wavelet", "Wavelet probe"),
                       ("residual", "Residual probe"), ("dct", "DCT probe"), ("fft", "FFT probe")):
        rows.append((label, (x1a[f"F4_{rep}"]["jpeg_q20"]["delta_auc"], BLUE)))
    rows.append(("DFDM (five generators)", None))
    for fam, label in (("D-F1", "ResNet-18"), ("D-EB4-F1", "EfficientNet-B4")):
        for s in SEEDS:
            rows.append((f"{label}, seed {s}", (DFDM["D_S1_delta_auc_q20"][f"{fam}_s{s}"], ORANGE)))
    step = 11.5
    top = 22
    height = top + len(rows) * step + 46
    svg = Svg(252, round(height))
    lo, hi = -0.6, 0.6
    ax = Axes(svg, 112, top, 132, len(rows) * step, 1, (0, len(rows)), pad=0)
    X = lambda v: ax.x + (v - lo) / (hi - lo) * ax.w
    for t in ticks(lo, hi, 0.3):
        svg.line(X(t), top, X(t), top + len(rows) * step, GRID if t else AXIS, 0.5 if t else 0.9)
        svg.text(X(t), top + len(rows) * step + 9, f"{t:g}".replace("-", "−"), 6.8, "middle", MUTED)
    for k, (label, entry) in enumerate(rows):
        y = top + (k + 0.5) * step
        if entry is None:
            svg.text(4, y + 2.5, label, 7.2, "start", INK, "bold")
            continue
        e, color = entry
        svg.text(108, y + 2.5, label, 6.9, "end", INK2)
        if e["ci95"]:
            svg.line(X(max(lo, e["ci95"][0])), y, X(min(hi, e["ci95"][1])), y, color, 1.2)
        svg.marker("circle", X(e["point"]), y, color, 2.8)
    base = top + len(rows) * step
    svg.text(X(0), base + 21, f"Δ{sub('AUC')}(Q20) = S{sub('AUC, attr')} − S{sub('AUC, det')}", 7.2, "middle", INK2,
             raw=True)
    svg.text(X(lo), base + 33, "← attr. lost faster", 6.2, "start", MUTED, italic=True)
    svg.text(X(hi), base + 33, "det. lost faster →", 6.2, "end", MUTED, italic=True)
    svg.text(4, 12, "Point and 95% group-bootstrap interval", 6.8, "start", MUTED, italic=True)
    return svg


def fig_e10():
    svg = Svg(252, 196)
    series = (("F1", BLUE, None, "circle"), ("F2", ORANGE, None, "circle"), ("F3", AQUA, None, "circle"),
              ("EB4-F1", BLUE, "3.5,2", "square"), ("EB4-F2", ORANGE, "3.5,2", "square"))
    sm = E10["seed_means"]
    values = [sm[f][c]["detection_auc"] for f, *_ in series for c in JPEG]
    ylim = (0.5, max(0.75, math.ceil(max(values) / 0.05) * 0.05))
    ax = Axes(svg, 40, 40, 204, 124, len(JPEG) - 1, ylim)
    ax.frame(ticks(*ylim, 0.05), list(enumerate(JPEG_LABELS)), yfmt="{:.2f}", ylabel="Detection AUROC")
    ax.hline(0.5, "chance")
    for fam, color, dash, shape in series:
        ax.series([sm[fam][c]["detection_auc"] for c in JPEG], color, dash, shape)
    legend(svg, 40, 12, [("Reference", BLUE, None, None), ("JPEG aug.", ORANGE, None, None),
                         ("Paired", AQUA, None, None)], gap=10)
    legend(svg, 40, 25, [("ResNet-18", INK2, None, "circle"), ("EfficientNet-B4", INK2, "3.5,2", "square")], gap=10)
    svg.text(142, 190, "Additional JPEG quality (seed means)", 7.2, "middle", INK2)
    return svg


FIGURES = {
    "fig1_retention": (fig_retention, "Chance-adjusted retention of detection and attribution under additional JPEG "
                       "compression for reference-trained models. Top: balanced accuracy at the frozen validation "
                       "operating point (registered metric). Bottom: AUROC-based retention, threshold-free (attribution "
                       "AUROC is the mean one-vs-rest AUROC over fake videos). FF++ ResNet-18 rows are the registered "
                       "results; FF++ EfficientNet-B4 (X6) and FF++ AUROC curves (X1a) are post-freeze extensions. "
                       "DFDM curves are the separately registered study. Lines are seed means; whiskers span the three "
                       "seeds. Retention above 1 (DFDM detection at Q95–Q80) means BA rose above its reference value: "
                       "the reference-only DFDM models have frozen thresholds near 0, so their reference BA is "
                       "depressed while detection AUROC stays near 100%."),
    "fig2_granularity": (fig_granularity, "Retention by attribution granularity at JPEG Q60 and Q20 for the "
                         "reference-trained ResNet-18. FF++: detection, identity-swap vs reenactment family, Deepfakes "
                         "vs FaceSwap, four-way methods, Face2Face vs NeuralTextures (content-matched decomposition C3c). "
                         "DFDM: detection and five-way generator attribution. Deepfakes vs FaceSwap is retained best; "
                         "Face2Face vs NeuralTextures loses most within FF++, and the five closely related DFDM "
                         "generators fall to chance by Q60. Retention is chance-adjusted, so tasks with different "
                         "chance levels are comparable."),
    "fig3_confusion": (fig_confusion, "Attribution confusion matrices of the reference-trained ResNet-18 at the "
                       "reference condition and at JPEG Q20 (rows: true class; three seeds pooled). FF++: DF Deepfakes, "
                       "F2F Face2Face, FS FaceSwap, NT NeuralTextures. DFDM: FSW Faceswap, LW Lightweight, IAE, DFK "
                       "DFaker, H128 DFL-H128. At Q20, FF++ reenactment methods drift towards NT. In DFDM, each seed "
                       "collapses onto one or two generators regardless of the true class (seed 42 mostly DFL-H128, "
                       "seed 43 Lightweight and DFL-H128, seed 44 DFaker), so the pooled Q20 rows are nearly identical."),
    "fig4_delta_auc": (fig_delta_auc, "Threshold-free contrast Δ_AUC(Q20) = attribution minus detection AUROC "
                       "retention, with 95% group-bootstrap intervals. FF++ values are post-freeze extensions "
                       "(X1a, X7, X8); DFDM values are the registered secondary contrast D-S1. DFDM frozen probes are "
                       "omitted because their reference attribution is near chance."),
    "fig5_e10": (fig_e10, "External detection test E10: frozen FF++ detectors on the 518 official Celeb-DF v2 test "
                 "videos, AUROC under additional JPEG compression (seed means)."),
}


# ------------------------------------------------------------------------------------------- tables


def num(x, digits=1, scale=100):
    s = f"{scale * x:.{digits}f}"
    return "−" + s[1:] if s.startswith("-") else s


def ci(e, digits=1, scale=100):
    if e is None or e["point"] is None:
        return "undefined"
    if not e["ci95"]:
        return f"{num(e['point'], digits, scale)} (no interval)"
    return f"{num(e['point'], digits, scale)} [{num(e['ci95'][0], digits, scale)}, {num(e['ci95'][1], digits, scale)}]"


def msd(values, digits=1):
    values = np.asarray(values, dtype=float)
    return f"{num(values.mean(), digits)} ± {num(values.std(ddof=1), digits)}"


SECTION = "§"


def table_tex(caption, label, header, rows, wide, notes):
    def tex(s):
        s = str(s)
        for a, b in (("%", r"\%"), ("_AUC", "@AUC"), ("S_det", r"$S_\mathrm{det}$"), ("S_attr", r"$S_\mathrm{attr}$"),
                     ("Δ@AUC", r"$\Delta_\mathrm{AUC}$"), ("Δ", r"$\Delta$"), ("±", r"$\pm$"), ("−", r"$-$"),
                     ("†", r"\textsuperscript{\dag}"), ("‡", r"\textsuperscript{\ddag}"), ("→", r"$\to$"),
                     ("@AUC", r"\_AUC")):
            s = s.replace(a, b)
        return s

    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", r"\centering", r"\footnotesize", f"\\caption{{{tex(caption)}}}", f"\\label{{{label}}}",
             r"\setlength{\tabcolsep}{3.5pt}", f"\\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}", r"\toprule",
             " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    first = True
    for row in rows:
        if row[0] == SECTION:
            if not first:
                lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\textit{{{tex(row[1])}}}}} \\\\")
        else:
            lines.append(" & ".join(tex(c) for c in row) + r" \\")
        first = False
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {tex(notes)}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def table_md(caption, header, rows, notes):
    out = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
    for row in rows:
        if row[0] == SECTION:
            out.append(f"| *{row[1]}* |" + " |" * (len(header) - 1))
        else:
            out.append("| " + " | ".join(str(c) for c in row) + " |")
    if notes:
        out += ["", notes]
    return "\n".join(out) + "\n"


def tab_datasets():
    pm = TEST["exploratory"]["point_metrics"]["F1_s42"]["ref"]
    sup = DFDM["support"]
    rows = [
        ["FF++ C23 (official test)", "Main study", pm["videos"], pm["videos"] - pm["fake_videos"], pm["fake_videos"],
         "4 methods", TEST["bootstrap"]["groups"], "H.264 C23 mirror"],
        ["DFDM + Celeb-DF reals (test)", "Second dataset", sup["videos"], sup["real"], sup["fake"], "5 generators",
         sup["groups"], "Lossless H.264 4:4:4"],
        ["Celeb-DF v2 (official test)", "External detection (E10)", E10["videos"], E10["real"], E10["fake"], "—",
         E10["groups"], "As distributed"],
    ]
    return ("Test sets. Bootstrap groups are FF++ provenance groups, DFDM ordered identity-pair groups and Celeb-DF "
            "identity groups.", "tab:datasets",
            ["Dataset", "Role", "Videos", "Real", "Fake", "Attribution", "Groups", "Reference"], rows, True, None)


def tab_ffpp_curves():
    summary = TEST["exploratory"]["seed_summary"]
    eb4 = EXT["X6_efficientnet_b4"]["curves"]
    rows = []
    for metric, title in (("detection_ba", "Detection BA (%)"), ("attribution_ba", "Attribution BA (%)")):
        rows.append([SECTION, title])
        for fam, label in (("F1", "ResNet-18, reference (F1)"), ("F2", "ResNet-18, JPEG aug. (F2)"),
                           ("F3", "ResNet-18, paired (F3)")):
            rows.append([label] + [msd(summary[fam][c][metric]["values_by_seed"]) for c in JPEG])
        for fam, label in (("EB4-F1", "EfficientNet-B4, reference†"), ("EB4-F2", "EfficientNet-B4, JPEG aug.†")):
            rows.append([label] + [msd([eb4[f"{fam}_s{s}"][c][metric]["point"] for s in SEEDS]) for c in JPEG])
    return ("FF++ test balanced accuracy under additional JPEG compression, mean ± SD over three seeds.",
            "tab:ffpp-curves", ["Model"] + list(JPEG_LABELS), rows, True,
            "Detection uses each model's frozen validation threshold; attribution is four-way on fake videos "
            "(chance 25%). † Post-freeze extension X6.")


def tab_ffpp_primary():
    iv, p1 = TEST["model_intervals"], TEST["primary_contrast"]["P1_RGB"]["per_seed"]
    x1a, eb4 = EXT["X1a_threshold_free"], EXT["X6_efficientnet_b4"]["P1_rule_at_q20"]["per_seed"]
    rows = []
    for s in SEEDS:
        q20 = iv[f"F1_s{s}"]["jpeg_q20"]
        rows.append([f"Seed {s}", ci(q20["detection_retention"], 2, 1), ci(q20["attribution_retention"], 2, 1),
                     ci(p1[f"F1_s{s}"], 2, 1), ci(x1a[f"F1_s{s}"]["jpeg_q20"]["delta_auc"], 2, 1),
                     ci(eb4[f"EB4-F1_s{s}"], 2, 1), ci(x1a[f"EB4-F1_s{s}"]["jpeg_q20"]["delta_auc"], 2, 1)])
    x3 = EXT["X3_seed_pooled"]
    rows.append(["Pooled†", "", "", ci(x3["F1_delta_q20"], 2, 1), "", ci(x3["EB4F1_delta_q20"], 2, 1), ""])
    return ("FF++ primary contrast at JPEG Q20 for reference-trained models (95% provenance-group bootstrap "
            "intervals). Registered reading for ResNet-18: attribution retained more than detection.",
            "tab:ffpp-primary",
            ["", "R18 S_det", "R18 S_attr", "R18 Δ", "R18 Δ_AUC†", "EB4 Δ†", "EB4 Δ_AUC†"], rows, True,
            "Δ = S_attr − S_det at the frozen operating point; Δ_AUC uses AUROC for both tasks. † Post-freeze "
            "extensions (X1a, X3, X6); the pooled row resamples training seeds and groups.")


def tab_dfdm_curves():
    sm = DFDM["seed_means"]
    rows = []
    for metric, title in (("attribution_ba", "Attribution BA (%), chance 20%"), ("detection_ba", "Detection BA (%)"),
                          ("detection_auc", "Detection AUROC (%)")):
        rows.append([SECTION, title])
        for fam, label in (("D-F1", "ResNet-18, reference"), ("D-F2", "ResNet-18, JPEG aug."),
                           ("D-EB4-F1", "EfficientNet-B4, reference"), ("D-EB4-F2", "EfficientNet-B4, JPEG aug.")):
            rows.append([label] + [f"{num(sm[fam][c][metric][0])} ± {num(sm[fam][c][metric][1])}" for c in DF_COLS])
    return ("DFDM test results, mean ± SD over three seeds. JPEG columns add compression to the lossless reference; "
            "CRF columns re-encode the whole video with DFDM's x264 settings.", "tab:dfdm-curves",
            ["Model"] + list(DF_LABELS), rows, True,
            "Detection results carry the compression-history caveat of Table~\\ref{tab:controls}.")


def tab_dfdm_primary():
    p1, s1 = DFDM["D_P1"], DFDM["D_S1_delta_auc_q20"]
    rows = [[f"Seed {s}", ci(p1["D-F1"]["per_seed"][f"D-F1_s{s}"], 2, 1), ci(s1[f"D-F1_s{s}"], 2, 1),
             ci(p1["D-EB4-F1"]["per_seed"][f"D-EB4-F1_s{s}"], 2, 1), ci(s1[f"D-EB4-F1_s{s}"], 2, 1)] for s in SEEDS]
    rows.append(["Registered reading", p1["D-F1"]["reading"], "", p1["D-EB4-F1"]["reading"], ""])
    return ("DFDM primary contrast D-P1 at JPEG Q20 for reference-trained models (95% identity-group bootstrap "
            "intervals).", "tab:dfdm-primary", ["", "R18 Δ", "R18 Δ_AUC", "EB4 Δ", "EB4 Δ_AUC"], rows, True,
            "A direction is claimed only when all three seed intervals exclude zero with the same sign.")


def tab_controls():
    s6 = TEST["secondary"]["S6_shortcut_and_content"]
    names = {"fft": "FFT", "rgb": "Pooled RGB", "dct": "DCT", "wavelet": "Wavelet", "residual": "Residual", "dinov2": "DINOv2"}
    rows = [[SECTION, "FF++ (chance 25% / 50%)"]]
    for rep, label in names.items():
        e = s6["C3a_border"][rep]
        flags = [k.split("_")[0] for k, v in e["material_flag_ref"].items() if v]
        rows.append([f"C3a face-masked {label}", ci(e["ref"]["attribution_ba"]), ci(e["ref"]["detection_ba"]),
                     ", ".join(flags) or "none"])
    meta = s6["C3b_metadata"]
    rows.append(["C3b metadata/geometry", ci(meta["attribution_ba"]), ci(meta["detection_ba"]),
                 ", ".join(k.split("_")[0] for k, v in meta["material_flag"].items() if v) or "none"])
    c1, c2 = DFDM["D_C1_compression_history"], DFDM["D_C2_face_masked"]
    rows.append([SECTION, "DFDM (chance 20% / 50%)"])
    rows.append(["C1 compression history (block DCT)", "—", ci(c1["detection_ba_ref"]),
                 "detection" if c1["material"] else "none"])
    rows.append(["C2 face-masked", ci(c2["attribution_ba"]), ci(c2["detection_ba"]), "none"])
    return ("Shortcut controls at the reference condition: test BA (%) with 95% bootstrap intervals.", "tab:controls",
            ["Control", "Attribution BA", "Detection BA", "Material"], rows, False,
            "FF++ flags: lower bound above 30% (attribution) or 55% (detection). DFDM C1 flag: lower bound above "
            "60%. C1 is detection-only.")


def tab_augmentation():
    s2 = TEST["secondary"]["S2_training"]["per_seed"]
    x6 = EXT["X6_efficientnet_b4"]
    s3 = DFDM["D_S3_augmentation"]
    rows = [[SECTION, "FF++, ResNet-18 (F2 − F1)"]]
    for s in SEEDS:
        d = s2[f"seed{s}"]["F2_minus_F1"]
        rows.append([f"Seed {s}", ci(d["ref"]["attribution_ba"]), ci(d["jpeg_q20"]["attribution_ba"]),
                     ci(d["ref"]["detection_ba"]), ci(d["jpeg_q20"]["detection_ba"])])
    rows.append([SECTION, "FF++, EfficientNet-B4 (X6)†"])
    for s in SEEDS:
        d = x6["F2_minus_F1_q20"][f"seed{s}"]
        ref = {m: x6["curves"][f"EB4-F2_s{s}"]["ref"][m]["point"] - x6["curves"][f"EB4-F1_s{s}"]["ref"][m]["point"]
               for m in ("attribution_ba", "detection_ba")}
        rows.append([f"Seed {s}", f"{num(ref['attribution_ba'])}‡", ci(d["attribution_ba"]),
                     f"{num(ref['detection_ba'])}‡", ci(d["detection_ba"])])
    for prefix, title in (("F2_minus_F1", "DFDM, ResNet-18"), ("EB4F2_minus_EB4F1", "DFDM, EfficientNet-B4")):
        rows.append([SECTION, title])
        for s in SEEDS:
            d = s3[f"{prefix}_seed{s}"]
            rows.append([f"Seed {s}", ci(d["ref"]["attribution_ba"]), ci(d["jpeg_q20"]["attribution_ba"]),
                         ci(d["ref"]["detection_ba"]), ci(d["jpeg_q20"]["detection_ba"])])
    return ("Effect of JPEG augmentation: augmented minus reference-trained BA, in percentage points.",
            "tab:augmentation", ["", "Attr. Ref", "Attr. Q20", "Det. Ref", "Det. Q20"], rows, True,
            "† Post-freeze extension. ‡ Point difference only; no interval was computed.")


def tab_e10():
    rows = []
    for fam, label in (("F1", "ResNet-18, reference (F1)"), ("F2", "ResNet-18, JPEG aug. (F2)"),
                       ("F3", "ResNet-18, paired (F3)"), ("EB4-F1", "EfficientNet-B4, reference"),
                       ("EB4-F2", "EfficientNet-B4, JPEG aug.")):
        cells = []
        for c in JPEG:
            vals = [E10["models"][f"{fam}_s{s}"][c]["detection_auc"]["point"] for s in SEEDS]
            assert np.isclose(np.mean(vals), E10["seed_means"][fam][c]["detection_auc"])
            cells.append(msd(vals))
        rows.append([label] + cells)
    return (f"External detection test E10: frozen FF++ detectors on the {E10['videos']} official Celeb-DF v2 test "
            "videos, AUROC (%), mean ± SD over three seeds.", "tab:e10", ["FF++ model"] + list(JPEG_LABELS), rows, True,
            "FF++ comparison register Amendment 6; models, thresholds and preprocessing unchanged.")


def tab_codec():
    models = EXT["X5_codec"]["models"]
    rows = []
    for metric, title in (("detection_ba", "Detection BA (%)"), ("attribution_ba", "Attribution BA (%)")):
        rows.append([SECTION, title])
        for fam, label in (("F1", "ResNet-18, reference"), ("F2", "ResNet-18, JPEG aug."), ("F3", "ResNet-18, paired"),
                           ("EB4-F1", "EfficientNet-B4, reference"), ("EB4-F2", "EfficientNet-B4, JPEG aug.")):
            rows.append([label] + [num(mean([models[f"{fam}_s{s}"][c][metric]["point"] for s in SEEDS]))
                                   for c in CODECS])
    dist = EXT["X5_codec"]["distortion"]
    rows.append([SECTION, "Distortion"])
    rows.append(["Crop PSNR vs. reference (dB)"] + [f"{dist[c]['mean_crop_psnr_db']:.1f}" for c in CODECS])
    rows.append(["Bitrate (kbps)"] + [f"{dist[c]['mean_kbps']:.0f}" for c in CODECS])
    return ("FF++ whole-video re-encoding (post-freeze extension X5): test BA, seed means. Column headers give "
            "codec and CRF.", "tab:codec", ["Model"] + list(CODEC_LABELS), rows, True,
            "Equal CRF across codecs is not equal distortion; compare by PSNR and bitrate.")


def tab_encoders():
    curves, x1a = JEPA["curves"], EXT["X1a_threshold_free"]
    delta_auc = {"F4_dinov2": x1a["F4_dinov2"]["jpeg_q20"]["delta_auc"],
                 "X7-REF": CLIP["X7_REF_delta_auc"]["jpeg_q20"]["delta_auc"],
                 "X8-REF": JEPA["X8_REF_delta_auc"]["jpeg_q20"]["delta_auc"]}
    rows = []
    for name, label in (("F4_dinov2", "DINOv2 ViT-S/14 (F4)"), ("X7-REF", "CLIP ViT-L/14†"),
                        ("X8-REF", "I-JEPA ViT-H/14‡")):
        c = curves[name]
        rows.append([label, num(c["ref"]["detection_ba"]["point"]), num(c["ref"]["attribution_ba"]["point"]),
                     num(c["jpeg_q20"]["detection_ba"]["point"]), num(c["jpeg_q20"]["attribution_ba"]["point"]),
                     ci(c["jpeg_q20"]["attribution_minus_detection_retention"], 2, 1), ci(delta_auc[name], 2, 1)])
    return ("Frozen pretrained encoders with ridge heads on FF++ test, trained at the reference condition: BA (%) "
            "and the Q20 contrasts with 95% group-bootstrap intervals.", "tab:encoders",
            ["Encoder", "Det. ref", "Attr. ref", "Det. Q20", "Attr. Q20", "Δ", "Δ_AUC†"], rows, True,
            "Δ is at the frozen validation operating point; Δ_AUC is threshold-free. DINOv2 values except Δ_AUC are "
            "registered. † Post-freeze extension (X1a, X7). ‡ Post-freeze extension X8, designed after all other "
            "results; its Q20 detection BA reflects a collapsed threshold, so Δ_AUC is the better reading.")


TABLES = {
    "tab1_datasets": tab_datasets, "tab2_ffpp_curves": tab_ffpp_curves, "tab3_ffpp_primary": tab_ffpp_primary,
    "tab4_dfdm_curves": tab_dfdm_curves, "tab5_dfdm_primary": tab_dfdm_primary, "tab6_controls": tab_controls,
    "tab7_augmentation": tab_augmentation, "tab8_e10": tab_e10, "tab9_codec": tab_codec, "tab10_encoders": tab_encoders,
}


# ------------------------------------------------------------------------------------------- output


def find_browser():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(p).exists():
            return p
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def render(svg_text, width, height, pdf, png, browser, tmp):
    html = tmp / (pdf.stem + ".html")
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>@page{{size:{width}pt {height}pt;margin:0}}"
                    f"html,body{{margin:0;padding:0;background:#fff;width:{width}pt;height:{height}pt;overflow:hidden}}"
                    f"svg{{display:block}}</style></head><body>{svg_text}</body></html>", encoding="utf-8")
    common = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
              f"--user-data-dir={tmp / 'profile'}"]
    subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, timeout=180, capture_output=True)
    subprocess.run(common + [f"--screenshot={png}", f"--window-size={math.ceil(width * 4 / 3)},{math.ceil(height * 4 / 3)}",
                             "--force-device-scale-factor=3", html.as_uri()], check=True, timeout=180, capture_output=True)
    assert pdf.exists() and png.exists(), pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--render", action="store_true", help="also write PDF and PNG with a headless Edge/Chrome")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    check_records()
    figures, tables = OUT / "figures", OUT / "tables"
    targets = [figures / f"{n}.svg" for n in FIGURES] + [tables / f"{n}.tex" for n in TABLES] + [tables / "tables.md"]
    if args.render:
        targets += [figures / f"{n}.{ext}" for n in FIGURES for ext in ("pdf", "png")]
    existing = [str(p.relative_to(ROOT)) for p in targets if p.exists()]
    if existing and not args.force:
        raise SystemExit("Outputs already exist (use --force to overwrite):\n  " + "\n  ".join(existing))
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    browser = find_browser() if args.render else None
    if args.render and not browser:
        raise SystemExit("--render needs Microsoft Edge or Google Chrome")
    with tempfile.TemporaryDirectory() as tmp:
        for name, (build, _) in FIGURES.items():
            svg = build()
            text = svg.render()
            (figures / f"{name}.svg").write_text(text, encoding="utf-8")
            if browser:
                render(text, svg.width, svg.height, figures / f"{name}.pdf", figures / f"{name}.png", browser, Path(tmp))
            print(f"figure {name}: {svg.width} x {svg.height} pt")
    md = ["# Paper tables", "", "Generated by `experiments/make_paper_figures.py` from the `research/main_results_*.json` "
          "records. LaTeX versions (booktabs) are in this folder.", ""]
    for name, build in TABLES.items():
        caption, label, header, rows, wide, notes = build()
        (tables / f"{name}.tex").write_text(table_tex(caption, label, header, rows, wide, notes), encoding="utf-8")
        md += [f"## {name}", "", table_md(caption, header, rows, notes and notes.replace("Table~\\ref{tab:controls}", "the controls table"))]
        print(f"table {name}: {sum(r[0] != SECTION for r in rows)} rows")
    (tables / "tables.md").write_text("\n".join(md), encoding="utf-8")
    captions = ["# Figure captions (drafts)", ""] + [f"- **{n}**: {c}" for n, (_, c) in FIGURES.items()]
    (figures / "captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
