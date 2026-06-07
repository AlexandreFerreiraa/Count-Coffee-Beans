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

- `V_efetivo` — volume ocupado pelos grãos = **capacidade interna MEDIDA do pote
  (3223 mL) × fração preenchida `f_fill`** (o `f_fill < 1` desconta o headspace
  vazio do pescoço). Usar o volume medido reduz muito a incerteza geométrica.
- `phi` — **fração de empacotamento** (random packing); o complemento é o ar
  entre os grãos.
- `v_grão`, `m_grão` — volume/massa de um grão; `rho_bulk` — densidade a granel.

> **Grãos torrados.** As propriedades (densidades, dimensões) usam faixas de
> literatura para café **torrado** (poroso e mais leve a granel que o verde).

Toda quantidade incerta é uma **variável aleatória**; a incerteza é propagada
por **Monte Carlo**.

## Modelos + ensemble

Para não depender de um único método (evitar viés de análise única):

| Modelo | Fórmula | Âncora |
|---|---|---|
| **A** Volumétrico | `V·phi/v_grão` | `phi` de literatura + massa experimental |
| **B** Massa/Bulk | `V·rho_bulk/m_grão` | densidade a granel + massa experimental |
| **C** Estereologia/Visão | `n_V·V` | densidade de grãos na parede (visão computacional) |
| **D** Forma+Packing | `V·phi(aspecto)/v_geo` | geometria do grão + física de empacotamento (esferoides) |
| **G** Gravimétrico *(opcional)* | `(M_cheio − M_vazio)/m_grão` | **pesagem do pote** (padrão-ouro) |

> **Para o melhor resultado possível:** pese o pote cheio e preencha
> `M_FULL_JAR_G` em `beanest/config.py`. Isso ativa o modelo **G** (gravimétrico),
> que vira uma quase-medição direta e domina o ensemble. A massa do pote vazio
> (1240 g) já está configurada.

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

Edite `beanest/config.py`. Com o volume do pote já medido, as maiores alavancas
agora (ver Sobol no relatório) são as **propriedades do grão**: `s_lin` (tamanho
real vs. experimento), `rho_app` e `rho_bulk` (densidades do torrado). Em ordem
de impacto:

1. **Pese o pote cheio** e preencha `M_FULL_JAR_G` → ativa o modelo gravimétrico
   (de longe o mais preciso).
2. **Pese/meça ~50 grãos do pote** → fixa `s_lin` e a massa por grão.
3. **Meça a altura do headspace** (espaço vazio do pescoço) → estreita `f_fill`.
4. **Fotos com escala** (régua/moeda) → melhoram a visão computacional.

Reduzir o `sd` de um parâmetro estreita o IC; mudar o `mean` desloca o palpite.

## Saídas (`outputs/`)

- `report.html` — relatório completo (abra no navegador).
- `report.md` — versão Markdown.
- `fig_*.svg` — densidades, forest plot, Sobol, convergência.
- `samples.csv` — amostras de cada modelo + ensemble (para análise externa).
