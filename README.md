# Coffee Bean Estimator (`beanest`)

Estimativa estatística do **número de grãos de café** dentro de um pote
cilíndrico de vidro, a partir de fotos e de um dado experimental
(**89 grãos = 11 g**). O resultado **não** é um número pontual: é uma
**distribuição** com **intervalo de credibilidade de 95%** e um **melhor palpite**.

> **Núcleo sem dependências.** `python run.py` roda só com a biblioteca padrão
> do Python (>= 3.10) e gera um relatório HTML autocontido com gráficos SVG.
> `numpy`/`opencv`/`matplotlib` são **opcionais** (visão computacional e PNGs).

---

## TL;DR — como rodar

```bash
cd coffee-bean-estimator
python run.py
# abra outputs/report.html no navegador
```

Com visão computacional nas suas fotos (opcional):

```bash
pip install -r requirements.txt          # numpy + opencv + matplotlib
# coloque as 3 fotos em ./images e ajuste calib.json
python run.py --images ./images --calib calib.json
```

Opções úteis: `--samples 120000` (mais amostras), `--no-sobol` (mais rápido),
`--seed 42`, `--outdir resultados`.

---

## A ideia matemática

A identidade física central é:

```
N = V_efetivo · phi / v_grão = V_efetivo · rho_bulk / m_grão
```

- `V_efetivo` — volume da **coluna de grãos** (cilindro do corpo do pote vezes a
  altura dos grãos; usar a altura da coluna já desconta o vazio do pescoço).
- `phi` — **fração de empacotamento** (random packing); o complemento é o ar
  entre os grãos.
- `v_grão`, `m_grão` — volume/massa de um grão; `rho_bulk` — densidade a granel.

Toda quantidade incerta é uma **variável aleatória**; a incerteza é propagada
por **Monte Carlo**.

## Quatro modelos semi-independentes + ensemble

Para não depender de um único método (evitar viés de análise única):

| Modelo | Fórmula | Âncora |
|---|---|---|
| **A** Volumétrico | `V·phi/v_grão` | `phi` de literatura + massa experimental |
| **B** Massa/Bulk | `V·rho_bulk/m_grão` | densidade a granel + massa experimental |
| **C** Estereologia/Visão | `n_V·V` | densidade de grãos na parede (visão computacional) |
| **D** Forma+Packing | `V·phi(aspecto)/v_geo` | geometria do grão + física de empacotamento (esferoides) |

O **ensemble (E)** combina as quatro distribuições por **pool linear ponderado**
(acomoda divergências) e **pool logarítmico** (consenso onde concordam). Os pesos
combinam **precisão** (1/variância) e **consistência** (proximidade ao consenso
robusto).

## Comparação e diagnósticos

- **Concordância**: coeficiente de sobreposição, Bhattacharyya, divergência de
  Jensen-Shannon e teste KS (subamostrado).
- **Sensibilidade**: índices de **Sobol** (1ª ordem e total) — quais parâmetros
  dominam a incerteza de cada modelo.
- **Convergência**: MCSE da mediana (bootstrap) e média acumulada.

## Estrutura

```
coffee-bean-estimator/
├── run.py                 # orquestrador (CLI)
├── calib.json             # calibração de escala da visão computacional
├── requirements.txt       # dependências OPCIONAIS
└── beanest/
    ├── config.py          # priors + proveniência de cada valor (EDITE AQUI)
    ├── mcstats.py         # Monte Carlo + estatística (stdlib)
    ├── geometry.py        # volume efetivo do pote
    ├── beans.py           # massa/volume/tamanho do grão
    ├── packing.py         # fração de empacotamento + phi(aspecto)
    ├── models.py          # modelos A,B,C,D + ensemble E
    ├── compare.py         # comparação, Sobol, convergência
    ├── vision.py          # visão computacional (opcional)
    ├── plotting.py        # gráficos SVG (+ matplotlib opcional)
    └── report.py          # relatório Markdown + HTML
```

## Como ajustar a precisão

Edite `beanest/config.py`. As maiores alavancas (ver Sobol no relatório) são as
**dimensões do pote** (`d_in_cm`, `h_beans_cm`). Se você medir o pote com uma
régua, estreite esses priors (reduza `sd`) e o IC encolhe bastante. Fotos com um
objeto de dimensão conhecida (régua/moeda) melhoram a calibração da visão
computacional.

## Saídas (`outputs/`)

- `report.html` — relatório completo (abra no navegador).
- `report.md` — versão Markdown.
- `fig_*.svg` — densidades, forest plot, Sobol, convergência.
- `samples.csv` — amostras de cada modelo + ensemble (para análise externa).
