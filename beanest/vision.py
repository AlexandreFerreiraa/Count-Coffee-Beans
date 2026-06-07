"""
vision — pipeline de visão computacional (OPCIONAL).

Objetivo: usar as fotos do pote como FONTE DE DADOS para os priors do modelo
(densidade areal de grãos na parede e dimensões dos grãos), reduzindo a
dependência de "chutes".

Requer: opencv-python (cv2) e numpy. Se ausentes, todo o módulo degrada para
None e o núcleo usa os priors do config (observações extraídas manualmente das
fotos). Importação é protegida para nunca quebrar o pipeline principal.

Técnicas (estado-da-arte clássico para contagem de grãos aglomerados):
  1. Pré-processamento: cinza -> CLAHE (equalização adaptativa) -> filtro
     bilateral (preserva bordas).
  2. Segmentação: limiar de Otsu/adaptativo -> morfologia (abertura) para
     remover ruído.
  3. Separação de grãos que se tocam: transformada de distância + WATERSHED
     com marcadores (separa blobs grudados — essencial em leitos densos).
  4. Medição: para cada região, ajuste de elipse -> eixos maior/menor (px).
  5. Calibração de escala: cm_por_px a partir de um diâmetro conhecido do pote
     (CLI/JSON). Converte px -> mm/cm.
  6. Saídas: distribuição de tamanho dos grãos (mm) e densidade areal
     (grãos/cm^2) numa região de interesse na parede do vidro.

Estas saídas podem SOBRESCREVER os priors `na_per_cm2`, `t_layer_cm`, `L_mm`,
`W_mm` em config, via beanest.report/run.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:  # importação protegida
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    _HAVE_CV = True
except Exception:  # pragma: no cover
    _HAVE_CV = False


def available() -> bool:
    return _HAVE_CV


def cm_per_px_from_diameter(jar_diameter_cm: float, jar_diameter_px: float) -> float:
    """Fator de calibração de escala a partir de um diâmetro conhecido."""
    return jar_diameter_cm / float(jar_diameter_px)


def _segment_beans(img_bgr):
    """Segmenta grãos e devolve regiões rotuladas via watershed."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # limiar de Otsu (grãos vs. fundo/sombra entre grãos)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers)
    return markers, n_markers


def analyze_image(
    path: str,
    cm_per_px: float,
    roi_frac: float = 0.6,
    min_area_px: int = 60,
) -> Optional[Dict]:
    """Analisa uma imagem e retorna observações de grãos.

    roi_frac: fração central da imagem usada como região de interesse (evita
    bordas do pote/fundo). cm_per_px: calibração de escala.
    """
    if not _HAVE_CV:
        return None
    img = cv2.imread(path)
    if img is None:
        return None

    h, w = img.shape[:2]
    # recorta ROI central na parede do pote
    cx0, cx1 = int(w * (0.5 - roi_frac / 2)), int(w * (0.5 + roi_frac / 2))
    cy0, cy1 = int(h * (0.5 - roi_frac / 2)), int(h * (0.5 + roi_frac / 2))
    roi = img[cy0:cy1, cx0:cx1]
    roi_area_px = roi.shape[0] * roi.shape[1]

    markers, _ = _segment_beans(roi)

    majors_mm: List[float] = []
    minors_mm: List[float] = []
    count = 0
    for label in range(2, markers.max() + 1):
        mask = (markers == label).astype("uint8")
        area = int(mask.sum())
        if area < min_area_px:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if len(c) < 5:
            continue
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(c)
        major = max(ax1, ax2) * cm_per_px * 10.0  # cm -> mm
        minor = min(ax1, ax2) * cm_per_px * 10.0
        # filtro de plausibilidade (grão de café: 5-16 mm maior; razão < 3.5)
        if 5.0 <= major <= 16.0 and minor > 0 and (major / minor) < 3.5:
            majors_mm.append(major)
            minors_mm.append(minor)
            count += 1

    if count == 0:
        return None

    roi_area_cm2 = roi_area_px * (cm_per_px ** 2)
    na_per_cm2 = count / roi_area_cm2

    def _mean(v):
        return sum(v) / len(v)

    def _sd(v):
        m = _mean(v)
        return (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5

    return {
        "path": os.path.basename(path),
        "n_beans_detected": count,
        "roi_area_cm2": roi_area_cm2,
        "na_per_cm2": na_per_cm2,
        "L_mm_mean": _mean(majors_mm),
        "L_mm_sd": _sd(majors_mm),
        "W_mm_mean": _mean(minors_mm),
        "W_mm_sd": _sd(minors_mm),
    }


def analyze_folder(folder: str, calib: Dict[str, Dict]) -> Optional[Dict]:
    """Analisa todas as imagens de uma pasta. `calib` mapeia nome_arquivo ->
    {"jar_diameter_cm":.., "jar_diameter_px":..} ou {"cm_per_px":..}.

    Retorna observações agregadas (média ponderada por nº de grãos) prontas para
    sobrescrever priors, ou None se CV indisponível/sem detecções.
    """
    if not _HAVE_CV:
        return None
    if not os.path.isdir(folder):
        return None

    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    results = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(exts):
            continue
        c = calib.get(fn) or calib.get("default") or {}
        if "cm_per_px" in c:
            cmpp = c["cm_per_px"]
        elif "jar_diameter_cm" in c and "jar_diameter_px" in c:
            cmpp = cm_per_px_from_diameter(c["jar_diameter_cm"], c["jar_diameter_px"])
        else:
            continue  # sem calibração não dá para medir em cm
        r = analyze_image(os.path.join(folder, fn), cmpp)
        if r:
            results.append(r)

    if not results:
        return None

    total = sum(r["n_beans_detected"] for r in results)
    def wavg(key):
        return sum(r[key] * r["n_beans_detected"] for r in results) / total

    return {
        "per_image": results,
        "n_images": len(results),
        "n_beans_total": total,
        "na_per_cm2_mean": wavg("na_per_cm2"),
        "L_mm_mean": wavg("L_mm_mean"),
        "W_mm_mean": wavg("W_mm_mean"),
    }


def observations_to_param_overrides(obs: Dict) -> Dict[str, dict]:
    """Converte observações da CV em sobrescritas de priors (specs)."""
    if not obs:
        return {}
    ov: Dict[str, dict] = {}
    if "na_per_cm2_mean" in obs:
        na = obs["na_per_cm2_mean"]
        ov["na_per_cm2"] = {"dist": "truncnormal", "mean": na, "sd": max(0.25 * na, 0.3),
                            "lo": max(0.5, 0.4 * na), "hi": 2.2 * na,
                            "units": "1/cm^2", "desc": "Densidade areal (medida por CV)",
                            "source": "Visão computacional (watershed) nas fotos do usuário."}
    if "L_mm_mean" in obs:
        L = obs["L_mm_mean"]
        ov["L_mm"] = {"dist": "truncnormal", "mean": L, "sd": max(0.12 * L, 0.6),
                      "lo": 0.6 * L, "hi": 1.4 * L, "units": "mm",
                      "desc": "Comprimento do grão (medido por CV)",
                      "source": "Visão computacional (ajuste de elipse)."}
    if "W_mm_mean" in obs:
        W = obs["W_mm_mean"]
        ov["W_mm"] = {"dist": "truncnormal", "mean": W, "sd": max(0.12 * W, 0.5),
                      "lo": 0.6 * W, "hi": 1.4 * W, "units": "mm",
                      "desc": "Largura do grão (medida por CV)",
                      "source": "Visão computacional (ajuste de elipse)."}
    return ov
