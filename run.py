#!/usr/bin/env python3
"""
run.py — orquestrador (CLI) da estimativa do número de grãos de café.

Uso típico (no VS Code / terminal):

    python run.py                      # roda tudo com os priors do config
    python run.py --samples 120000     # mais amostras de Monte Carlo
    python run.py --images ./images --calib calib.json   # usa visão computacional

Saídas em ./outputs/ :
    report.html   (relatório autocontido, abra no navegador)
    report.md     (versão Markdown)
    fig_*.svg     (figuras)
    samples.csv   (amostras de cada modelo + ensemble, para análise externa)

Sem dependências obrigatórias. matplotlib/opencv/scipy são usados se presentes.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from beanest import compare, models, report, vision
from beanest.config import N_SAMPLES, PARAMS, RANDOM_SEED


def _apply_overrides(overrides: dict):
    """Sobrescreve specs de PARAMS in-place (ex.: vindos da visão computacional)."""
    for nm, spec in overrides.items():
        PARAMS[nm] = spec


def main():
    ap = argparse.ArgumentParser(description="Estimativa do número de grãos de café num pote.")
    ap.add_argument("--samples", type=int, default=N_SAMPLES, help="amostras de Monte Carlo")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--images", default=None, help="pasta com as fotos do pote (CV opcional)")
    ap.add_argument("--calib", default=None, help="JSON de calibração de escala (CV)")
    ap.add_argument("--no-sobol", action="store_true", help="pula a análise de Sobol (mais rápido)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[beanest] semente={args.seed}  amostras={args.samples}")

    # 1) Visão computacional (opcional) -> sobrescreve priors
    vision_obs = None
    if args.images:
        if not vision.available():
            print("[beanest] AVISO: opencv/numpy ausentes; CV desativada. "
                  "Instale 'opencv-python numpy' para ativar.")
        else:
            calib = {}
            if args.calib and os.path.exists(args.calib):
                with open(args.calib, encoding="utf-8") as f:
                    calib = json.load(f)
            print(f"[beanest] analisando imagens em {args.images} ...")
            vision_obs = vision.analyze_folder(args.images, calib)
            if vision_obs:
                ov = vision.observations_to_param_overrides(vision_obs)
                _apply_overrides(ov)
                print(f"[beanest] CV ok: {vision_obs['n_beans_total']} grãos detectados; "
                      f"priors sobrescritos: {list(ov.keys())}")
            else:
                print("[beanest] CV não retornou observações (sem calibração ou sem detecções).")

    # 2) Modelos + ensemble
    print("[beanest] rodando os 4 modelos + ensemble ...")
    results = models.run_all(n=args.samples, seed=args.seed)

    # 3) Comparações
    print("[beanest] comparando modelos (overlap/JS/KS) ...")
    desc = compare.descriptive_table(results["samples"])
    pairwise = compare.pairwise_agreement(results["samples"])
    sanity = compare.cross_consistency_with_experiment(results["samples"])

    if args.no_sobol:
        sens = {k: {} for k in results["models"]}
    else:
        print("[beanest] análise de sensibilidade (Sobol) ...")
        sens = compare.sensitivity(results["models"])

    print("[beanest] diagnóstico de convergência ...")
    conv = compare.convergence(results["ensemble_linear"])

    # 4) CSV de amostras
    os.makedirs(args.outdir, exist_ok=True)
    _dump_samples_csv(results, os.path.join(args.outdir, "samples.csv"))

    # 5) Relatório
    print("[beanest] gerando relatório ...")
    paths = report.build(results, desc, pairwise, sens, conv, sanity, vision_obs, args.outdir)

    es = __import__("beanest.mcstats", fromlist=["summary"]).summary(results["ensemble_linear"])
    print("\n" + "=" * 60)
    print(f"  MELHOR PALPITE : {es['median']:,.0f} grãos".replace(",", "."))
    print(f"  IC 95%         : [{es['p2.5']:,.0f} ; {es['p97.5']:,.0f}]".replace(",", "."))
    print(f"  MCSE da mediana: ±{conv['median_mcse']:,.0f}".replace(",", "."))
    print("=" * 60)
    print(f"[beanest] relatório: {paths['html']}")
    print(f"[beanest] concluído em {time.time()-t0:.1f}s")


def _dump_samples_csv(results, path):
    keys = list(results["samples"].keys())
    n = len(results["ensemble_linear"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys + ["ensemble"]) + "\n")
        cols = [results["samples"][k] for k in keys] + [results["ensemble_linear"]]
        for i in range(n):
            f.write(",".join(f"{c[i]:.1f}" for c in cols) + "\n")


if __name__ == "__main__":
    main()
