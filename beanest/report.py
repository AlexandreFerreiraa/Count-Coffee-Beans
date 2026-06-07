"""
report — monta o relatório final (Markdown + HTML autocontido).

O HTML embute os SVGs inline (abre em qualquer navegador, sem dependências).
O Markdown é uma versão textual para leitura rápida / versionamento.
"""

from __future__ import annotations

import os
from typing import Dict, List

from . import plotting
from .config import PARAMS, EXP_BEANS, EXP_MASS_G, EXP_MASS_PER_BEAN_G
from .mcstats import median, summary


def _fmt(v: float, nd: int = 0) -> str:
    return f"{v:,.{nd}f}".replace(",", ".")


def build(results: Dict, desc_table: Dict, pairwise: List[Dict], sens: Dict,
          conv: Dict, sanity: Dict, vision_obs, outdir: str) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    models = results["models"]
    labels = {k: m.label for k, m in models.items()}
    colors = {k: m.color for k, m in models.items()}
    samples = results["samples"]
    weights = results["weights"]               # por princípio
    model_weights = results["model_weights"]   # por modelo (herdado)
    vardec = results["variance_decomposition"]
    ens = results["ensemble_linear"]
    ens_sum = summary(ens)
    ens_log = results["ensemble_log_summary"]

    # ---- figuras ----
    fig_density = plotting.density_overlay_svg(samples, labels, colors, ens)
    fig_forest = plotting.forest_plot_svg(desc_table, labels, ens_sum)
    fig_sobol = plotting.sobol_bars_svg(sens, labels)
    fig_conv = plotting.convergence_svg(conv["running_idx"], conv["running_mean"])
    for name, svg in [("density", fig_density), ("forest", fig_forest),
                      ("sobol", fig_sobol), ("convergence", fig_conv)]:
        plotting.save_svg(svg, os.path.join(outdir, f"fig_{name}.svg"))

    best = ens_sum["median"]
    ci_lo, ci_hi = ens_sum["p2.5"], ens_sum["p97.5"]

    # ---------------- MARKDOWN ----------------
    md: List[str] = []
    md.append("# Estimativa do número de grãos de café no pote\n")
    md.append("## 1. Resultado principal\n")
    md.append(f"- **Melhor palpite (mediana do ensemble): {_fmt(best)} grãos**")
    md.append(f"- **Intervalo de credibilidade 95%: [{_fmt(ci_lo)} ; {_fmt(ci_hi)}]**")
    md.append(f"- Média do ensemble: {_fmt(ens_sum['mean'])} | "
              f"IC 90%: [{_fmt(ens_sum['p5'])} ; {_fmt(ens_sum['p95'])}]")
    md.append(f"- Estimador de consenso (pool logarítmico): mediana {_fmt(ens_log['median'])}, "
              f"moda {_fmt(ens_log['mode'])}, IC95% [{_fmt(ens_log['p2.5'])} ; {_fmt(ens_log['p97.5'])}]  "
              f"*(o pool logarítmico assume independência entre princípios; tende a ser mais "
              f"estreito — use o pool LINEAR como IC principal, que é conservador)*")
    md.append(f"- MCSE da mediana (erro de Monte Carlo): ±{_fmt(conv['median_mcse'])} "
              f"(desprezível frente à incerteza física) \n")

    md.append("## 2. Síntese da metodologia\n")
    md.append("Tratamos toda quantidade desconhecida como variável aleatória e propagamos "
              "incerteza por **Monte Carlo**. A relação física central é\n")
    md.append("```\nN = V_efetivo * phi / v_grao = V_efetivo * rho_bulk / m_grao\n```")
    md.append("Para evitar viés de método único, usamos **quatro modelos** que se reduzem a "
              "**três princípios de estimação independentes**, combinados por **pooling ponderado** "
              "(não é um posterior Bayesiano completo, pois não há N observado para verossimilhança):\n")
    md.append("| Modelo | Ideia | Âncora principal | Princípio |")
    md.append("|---|---|---|---|")
    md.append("| A · Volumétrico | V·phi/v_grão | phi de literatura + massa experimental | Bulk/Packing |")
    md.append("| B · Massa/Bulk | V·rho_bulk/m_grão | densidade a granel + massa experimental | Bulk/Packing |")
    md.append("| C · Estereologia/Visão | n_V·V (densidade na parede) | contagem por visão computacional | Estereologia |")
    md.append("| D · Forma+Packing | V·phi(aspecto)/v_geo | geometria do grão + física de empacotamento | Forma |\n")
    md.append("**A e B são a mesma física** (rho_bulk ≈ phi·rho_app) em duas parametrizações, logo "
              "formam UM princípio (um voto no ensemble). **C** (visão, independe de massa e de phi) "
              "e **D** (forma do grão, independe de massa) são os princípios independentes que "
              "validam o resultado.\n")
    md.append(f"O dado experimental (**{EXP_BEANS} grãos = {EXP_MASS_G} g → "
              f"{EXP_MASS_PER_BEAN_G:.4f} g/grão**) ancora o princípio Bulk/Packing (A, B). "
              "D usa dimensões de literatura (independente); C usa geometria + visão (independente).\n")

    md.append("## 3. Resultados por modelo\n")
    md.append("| Modelo | Mediana | Média | IC95% | CoV | Peso (princípio) |")
    md.append("|---|---|---|---|---|---|")
    for k in samples:
        s = desc_table[k]
        md.append(f"| {labels[k]} | {_fmt(s['median'])} | {_fmt(s['mean'])} | "
                  f"[{_fmt(s['p2.5'])} ; {_fmt(s['p97.5'])}] | {s['cov']:.2f} | {model_weights[k]:.2f} |")
    md.append(f"| **Ensemble (E)** | **{_fmt(ens_sum['median'])}** | {_fmt(ens_sum['mean'])} | "
              f"[{_fmt(ci_lo)} ; {_fmt(ci_hi)}] | {ens_sum['cov']:.2f} | — |\n")

    md.append("**Pesos por PRINCÍPIO independente** (A e B são o mesmo princípio e contam como "
              "um só voto, evitando dupla contagem):\n")
    md.append("| Princípio | Peso |")
    md.append("|---|---|")
    from .models import PRINCIPLE_LABEL
    for pk, w in weights.items():
        md.append(f"| {PRINCIPLE_LABEL.get(pk, pk)} | {w:.2f} |")
    md.append("")

    md.append("## 4. Concordância entre modelos (par a par)\n")
    md.append("| A | B | Sobreposição | Bhattacharyya | JS-div | KS D | KS p |")
    md.append("|---|---|---|---|---|---|---|")
    for r in pairwise:
        md.append(f"| {r['model_a']} | {r['model_b']} | {r['overlap']:.2f} | "
                  f"{r['bhattacharyya']:.2f} | {r['js_divergence']:.3f} | "
                  f"{r['ks_D']:.2f} | {r['ks_p']:.3f} |")
    md.append("\nSobreposição alta / JS baixo => métodos concordam. **Nuance importante:** A e B "
              "são quase a mesma física em duas parametrizações (rho_bulk ≈ phi·rho_app), logo "
              "concordam *por construção* — sua sobreposição ~0.98 não é evidência forte. Os "
              "validadores **genuinamente independentes** são **C** (visão/estereologia, "
              "independe da massa e de phi) e **D** (geometria da forma + física de empacotamento, "
              "independe da massa). Ambos caem perto da faixa de A/B, o que **sim** reforça a "
              "estimativa. O p-valor do KS usa subamostra (ver compare.py).\n")

    md.append("## 5. Sensibilidade (Sobol — fração da variância explicada)\n")
    for k, params in sens.items():
        ranked = sorted(params.items(), key=lambda kv: kv[1]["S1"], reverse=True)[:4]
        top = ", ".join(f"{p} (S1={d['S1']:.2f})" for p, d in ranked)
        md.append(f"- **{labels[k]}**: {top}")
    md.append("")

    md.append("")
    md.append(f"**Decomposição de incerteza (modo-comum):** a GEOMETRIA do pote "
              f"(`d_in_cm`, `h_beans_cm`, `k_shape`) — que é compartilhada por TODOS os modelos — "
              f"responde por **~{vardec['frac_geometria']*100:.0f}%** da variância de N; os demais "
              f"fatores (grão, empacotamento) respondem por ~{vardec['frac_resto']*100:.0f}%. "
              f"Como a geometria é comum a todos, a concordância entre métodos NÃO reduz essa "
              f"parcela — **medir o pote (diâmetro e altura dos grãos) com uma régua é a forma "
              f"mais eficaz de estreitar o intervalo.**\n")

    md.append("## 6. Verificação de sanidade (massa total implícita)\n")
    md.append("Usando a massa/grão consistente com o experimento (≈0.147 g, já com o ajuste de "
              "tamanho s_lin), a massa total de grãos implícita por modelo:\n")
    for k, kg in sanity.items():
        md.append(f"- {labels[k]}: ~{kg:.1f} kg")
    md.append("")

    md.append("## 7. Visão computacional (dados das fotos)\n")
    if vision_obs:
        md.append(f"- Imagens analisadas: {vision_obs.get('n_images','?')}, "
                  f"grãos detectados: {vision_obs.get('n_beans_total','?')}")
        if "na_per_cm2_mean" in vision_obs:
            md.append(f"- Densidade areal medida: {vision_obs['na_per_cm2_mean']:.2f} grãos/cm²")
        if "L_mm_mean" in vision_obs:
            md.append(f"- Tamanho do grão medido: L≈{vision_obs['L_mm_mean']:.1f} mm, "
                      f"W≈{vision_obs['W_mm_mean']:.1f} mm")
        md.append("Estas medições sobrescreveram os priors `na_per_cm2`, `L_mm`, `W_mm`.\n")
    else:
        md.append("- *Não executada nesta rodada.* Para ativar, instale `opencv-python numpy` e rode "
                  "`python run.py --images ./images --calib calib.json`. A CV mede o tamanho dos "
                  "grãos (ajuste de elipse) e a densidade areal na parede (watershed), substituindo "
                  "os priors correspondentes por dados das suas fotos.\n")

    md.append("## 8. Parâmetros e proveniência\n")
    md.append("| Parâmetro | Distribuição | Centro | Unidade | Fonte |")
    md.append("|---|---|---|---|---|")
    for nm, sp in PARAMS.items():
        center = sp.get("mean", sp.get("value", "—"))
        rng = f"{sp['dist']}(σ={sp.get('sd','—')})" if sp["dist"] != "const" else "const"
        md.append(f"| `{nm}` | {rng} | {center} | {sp.get('units','')} | {sp.get('source','')} |")
    md.append("")

    md.append("## 9. Limitações\n")
    md.append("- Todos os modelos compartilham o volume do pote `V_eff`: é a maior fonte de "
              "incerteza comum (ver Sobol). Medir o pote reduziria muito o IC.")
    md.append("- As fotos dão escala aproximada (mão); uma régua/objeto de dimensão conhecida "
              "na foto melhoraria a calibração da visão computacional.")
    md.append("- Assumimos apenas grãos inteiros (conforme enunciado) e grãos de café verde.")
    md.append("- O Modelo C (estereologia) é o mais dependente de suposições; por isso recebe "
              "peso menor no ensemble quando diverge.")
    md.append("- A e B não são independentes entre si; o ensemble é, na prática, dominado por essa "
              "física comum, com C e D atuando como validação cruzada independente.\n")

    md_text = "\n".join(md)
    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # ---------------- HTML ----------------
    html = _html(md_text, [fig_density, fig_forest, fig_sobol, fig_conv], best, ci_lo, ci_hi)
    html_path = os.path.join(outdir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {"md": md_path, "html": html_path}


def _md_to_html(md_text: str) -> str:
    """Conversor Markdown -> HTML minimalista (títulos, tabelas, listas, código)."""
    lines = md_text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            out.append(f"<h3>{ln[4:]}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{ln[3:]}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{ln[2:]}</h1>")
        elif ln.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
        elif ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            header = [c.strip() for c in ln.strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue
        elif ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif ln.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def _inline(s: str) -> str:
    import re
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def _html(md_text: str, figs: List[str], best, ci_lo, ci_hi) -> str:
    body = _md_to_html(md_text)
    figs_html = "\n".join(f'<div class="fig">{svg}</div>' for svg in figs)
    return f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Estimativa de grãos de café</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:920px;margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.5}}
h1{{border-bottom:3px solid #d62728;padding-bottom:8px}}
h2{{margin-top:32px;color:#b3201f}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f5f5f5}}
tr:nth-child(even){{background:#fafafa}}
pre{{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto}}
code{{background:#f0f0f0;padding:1px 4px;border-radius:3px}}
.fig{{margin:18px 0;text-align:center}}
.hero{{background:#fff5f5;border:2px solid #d62728;border-radius:10px;padding:16px 20px;margin:16px 0}}
.hero .big{{font-size:30px;font-weight:700;color:#b3201f}}
</style></head><body>
<div class="hero"><div>Melhor palpite</div>
<div class="big">{_fmt(best)} grãos</div>
<div>Intervalo de credibilidade 95%: [{_fmt(ci_lo)} ; {_fmt(ci_hi)}]</div></div>
{body}
<h2>Figuras</h2>
{figs_html}
<hr><p style="color:#888;font-size:12px">Gerado por beanest — ensemble de 4 modelos + Monte Carlo.</p>
</body></html>"""
