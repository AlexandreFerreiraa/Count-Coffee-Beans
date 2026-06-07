"""
plotting — figuras do relatório.

Estratégia de robustez: gera gráficos em SVG com Python puro (sempre disponível,
sem dependências). Os SVGs são embutidos diretamente no relatório HTML. Se
matplotlib estiver instalado, também salva PNGs de alta qualidade (opcional).

Gráficos:
  - density_overlay : densidades (KDE) dos modelos + ensemble
  - forest_plot     : mediana + IC95% por modelo e ensemble
  - sobol_bars      : índices de Sobol (S1) dos parâmetros dominantes
  - convergence_plot: média acumulada (diagnóstico de Monte Carlo)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from . import mcstats

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# Infra SVG mínima
# ---------------------------------------------------------------------------
class SVG:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts: List[str] = []

    def _e(self, s: str):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1, opacity=1.0):
        self._e(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    def line(self, x1, y1, x2, y2, stroke="#333", sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._e(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def polyline(self, pts: List[Tuple[float, float]], stroke="#1f77b4", sw=2, fill="none", opacity=1.0):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self._e(f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{sw}" opacity="{opacity}"/>')

    def polygon(self, pts: List[Tuple[float, float]], fill="#1f77b4", opacity=0.2, stroke="none"):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self._e(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>')

    def circle(self, x, y, r, fill="#333"):
        self._e(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"/>')

    def text(self, x, y, s, size=12, fill="#222", anchor="start", weight="normal"):
        s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self._e(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                f'font-family="Segoe UI, Arial, sans-serif" text-anchor="{anchor}" '
                f'font-weight="{weight}">{s}</text>')

    def render(self) -> str:
        body = "\n".join(self.parts)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
                f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>')


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v/1000:.1f}k"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# Gráficos (SVG)
# ---------------------------------------------------------------------------
def density_overlay_svg(samples: Dict[str, List[float]], labels: Dict[str, str],
                        colors: Dict[str, str], ensemble: List[float],
                        title: str = "Distribuições posteriores de N") -> str:
    W, H = 760, 420
    ml, mr, mt, mb = 60, 180, 40, 50
    pw, ph = W - ml - mr, H - mt - mb
    svg = SVG(W, H)
    svg.text(ml, 24, title, size=15, weight="bold")

    series = dict(samples)
    series["ENSEMBLE"] = ensemble
    colors = dict(colors)
    colors["ENSEMBLE"] = "#d62728"
    labels = dict(labels)
    labels["ENSEMBLE"] = "Ensemble (E)"

    all_vals = [v for s in series.values() for v in s]
    lo, hi = min(all_vals), max(all_vals)
    grid = mcstats._linspace(lo, hi, 200)
    dens = {k: mcstats.kde_on_grid(v, grid) for k, v in series.items()}
    dmax = max(max(d) for d in dens.values())

    def X(v): return ml + (v - lo) / (hi - lo) * pw
    def Y(d): return mt + ph - (d / dmax) * ph

    # eixos
    svg.line(ml, mt + ph, ml + pw, mt + ph, "#333", 1.5)
    svg.line(ml, mt, ml, mt + ph, "#333", 1.5)
    for q in range(0, 6):
        v = lo + (hi - lo) * q / 5
        x = X(v)
        svg.line(x, mt + ph, x, mt + ph + 5, "#333", 1)
        svg.text(x, mt + ph + 20, _fmt(v), size=11, anchor="middle")
    svg.text(ml + pw / 2, H - 8, "Número de grãos N", size=12, anchor="middle")

    for k in series:
        pts = [(X(g), Y(d)) for g, d in zip(grid, dens[k])]
        sw = 3 if k == "ENSEMBLE" else 1.8
        svg.polyline(pts, stroke=colors[k], sw=sw)

    # legenda
    ly = mt + 6
    for k in series:
        svg.line(ml + pw + 16, ly, ml + pw + 40, ly, colors[k], 3)
        med = mcstats.median(series[k])
        svg.text(ml + pw + 46, ly + 4, f"{labels[k]} (med {_fmt(med)})", size=11)
        ly += 22
    return svg.render()


def forest_plot_svg(stats: Dict[str, Dict[str, float]], labels: Dict[str, str],
                    ens_summary: Dict[str, float], title: str = "Mediana e IC 95% por modelo") -> str:
    rows = list(stats.keys()) + ["ENSEMBLE"]
    W, H = 760, 60 + 56 * len(rows)
    ml, mr, mt = 230, 90, 50
    pw = W - ml - mr
    svg = SVG(W, H)
    svg.text(20, 26, title, size=15, weight="bold")

    los = [stats[k]["p2.5"] for k in stats] + [ens_summary["p2.5"]]
    his = [stats[k]["p97.5"] for k in stats] + [ens_summary["p97.5"]]
    lo, hi = min(los), max(his)
    pad = 0.05 * (hi - lo)
    lo -= pad; hi += pad

    def X(v): return ml + (v - lo) / (hi - lo) * pw

    for tick in range(6):
        v = lo + (hi - lo) * tick / 5
        x = X(v)
        svg.line(x, mt, x, H - 30, "#eee", 1)
        svg.text(x, H - 12, _fmt(v), size=11, anchor="middle")

    y = mt + 20
    for k in rows:
        if k == "ENSEMBLE":
            med, l, h, col = ens_summary["median"], ens_summary["p2.5"], ens_summary["p97.5"], "#d62728"
            lab = "Ensemble (E)"
        else:
            s = stats[k]
            med, l, h, col = s["median"], s["p2.5"], s["p97.5"], "#444"
            lab = labels.get(k, k)
        svg.text(20, y + 4, lab, size=12, weight=("bold" if k == "ENSEMBLE" else "normal"))
        svg.line(X(l), y, X(h), y, col, 3)
        svg.line(X(l), y - 5, X(l), y + 5, col, 2)
        svg.line(X(h), y - 5, X(h), y + 5, col, 2)
        svg.circle(X(med), y, 5, col)
        svg.text(X(h) + 8, y + 4, _fmt(med), size=11, fill=col)
        y += 56
    return svg.render()


def sobol_bars_svg(sens: Dict[str, Dict[str, Dict[str, float]]], labels: Dict[str, str],
                   title: str = "Sensibilidade (Sobol S1) — top parâmetros") -> str:
    # agrega: para cada modelo, top 4 parâmetros por S1
    W = 760
    blocks = []
    for k, params in sens.items():
        ranked = sorted(params.items(), key=lambda kv: kv[1]["S1"], reverse=True)[:4]
        blocks.append((k, ranked))
    row_h = 26
    H = 50 + sum(40 + row_h * len(r) for _, r in blocks)
    ml = 180
    pw = W - ml - 80
    svg = SVG(W, int(H))
    svg.text(20, 26, title, size=15, weight="bold")
    y = 52
    for k, ranked in blocks:
        svg.text(20, y, labels.get(k, k), size=13, weight="bold")
        y += 20
        for pname, idx in ranked:
            s1 = max(0.0, min(1.0, idx["S1"]))
            svg.text(28, y + 13, pname, size=11)
            svg.rect(ml, y, pw, 16, fill="#eee")
            svg.rect(ml, y, pw * s1, 16, fill="#1f77b4")
            svg.text(ml + pw + 6, y + 13, f"{s1:.2f}", size=11)
            y += row_h
        y += 14
    return svg.render()


def convergence_svg(idx: List[int], run: List[float], title: str = "Convergência de Monte Carlo (média acumulada)") -> str:
    W, H = 760, 320
    ml, mr, mt, mb = 70, 30, 40, 50
    pw, ph = W - ml - mr, H - mt - mb
    svg = SVG(W, H)
    svg.text(ml, 26, title, size=15, weight="bold")
    lo, hi = min(run[len(run)//5:]), max(run[len(run)//5:])
    span = (hi - lo) or 1.0
    lo -= 0.3 * span; hi += 0.3 * span
    xmin, xmax = idx[0], idx[-1]

    def X(i): return ml + (i - xmin) / (xmax - xmin) * pw
    def Y(v): return mt + ph - (v - lo) / (hi - lo) * ph

    svg.line(ml, mt + ph, ml + pw, mt + ph, "#333", 1.5)
    svg.line(ml, mt, ml, mt + ph, "#333", 1.5)
    for t in range(5):
        v = lo + (hi - lo) * t / 4
        yv = Y(v)
        svg.line(ml - 4, yv, ml, yv, "#333", 1)
        svg.text(ml - 8, yv + 4, _fmt(v), size=10, anchor="end")
    final = run[-1]
    svg.line(ml, Y(final), ml + pw, Y(final), "#d62728", 1, dash="4,4")
    svg.polyline([(X(i), Y(v)) for i, v in zip(idx, run)], stroke="#1f77b4", sw=2)
    svg.text(ml + pw / 2, H - 12, "Nº de amostras de Monte Carlo", size=12, anchor="middle")
    return svg.render()


def save_svg(svg_str: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_str)
