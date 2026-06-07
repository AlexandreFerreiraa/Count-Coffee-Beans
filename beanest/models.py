"""
models — estimadores do número de grãos N e o ensemble Bayesiano.

Todos os estimadores compartilham a identidade física básica:

    N = (volume ocupado pelos grãos) * (fração de empacotamento) / (volume do grão)
      = V_eff * phi / v_bean
      = V_eff * rho_bulk / m_bean         (forma equivalente via densidade a granel)

Implementamos QUATRO rotas semi-independentes, de modo que vieses de uma não
contaminem as outras (mitiga o "viés de análise única"):

  A  Volumétrico        : V_eff * phi / v_bean        (phi de literatura; v_bean via massa exp.)
  B  Massa/Bulk         : V_eff * rho_bulk / m_bean    (densidade a granel de literatura; massa exp.)
  C  Estereologia/CV    : n_V * V_eff, n_V = (na/t)*c  (densidade na parede medida por visão)
  D  Forma + Packing    : V_eff * phi(forma) / v_geo   (geometria do grão; phi da razão de aspecto)

Dependência compartilhada: TODAS usam V_eff (volume do pote). Isso é inevitável
e fica explícito na análise de sensibilidade. Fora isso:
  - A e B usam o experimento de massa (89 grãos = 11 g);
  - C é ancorado em geometria + contagem por visão (independe de massa e de phi);
  - D é puramente geométrico de forma + física de empacotamento (independe de massa).

O ensemble (Modelo E) combina as 4 distribuições por:
  - pool linear ponderado (mistura) — acomoda divergência entre métodos;
  - pool logarítmico (produto de densidades) — "consenso" onde os métodos concordam.
Os pesos combinam PRECISÃO (1/variância) e CONSISTÊNCIA (proximidade ao
consenso robusto), normalizados.
"""

from __future__ import annotations

from typing import Dict, List

from . import beans, config, geometry, mcstats, packing
from .config import MODEL_PARAMS, PARAMS, N_SAMPLES
from .mcstats import RNG


# ---------------------------------------------------------------------------
# Estimador base
# ---------------------------------------------------------------------------
class Estimator:
    key: str = ""
    label: str = ""
    color: str = "#888888"

    def param_specs(self) -> Dict[str, dict]:
        return {nm: PARAMS[nm] for nm in MODEL_PARAMS[self.key]}

    def compute(self, p: Dict[str, List[float]]) -> List[float]:
        """N elemento a elemento dado um dict de parâmetros (vetorizado)."""
        raise NotImplementedError

    def sample(self, n: int, rng: RNG) -> List[float]:
        specs = self.param_specs()
        p = {nm: mcstats.sample_spec(spec, n, rng.spawn(i)) for i, (nm, spec) in enumerate(specs.items())}
        return self.compute(p)


# ---------------------------------------------------------------------------
# Modelo A — Volumétrico
# ---------------------------------------------------------------------------
class ModelA(Estimator):
    key = "A_volumetrico"
    label = "A · Volumétrico (phi literatura)"
    color = "#1f77b4"

    def compute(self, p):
        V = geometry.effective_volume_from_params(p)
        v_bean = beans.volume_exp_from_params(p)
        return [Vi * phi / vb for Vi, phi, vb in zip(V, p["phi"], v_bean)]


# ---------------------------------------------------------------------------
# Modelo B — Massa / Densidade a granel
# ---------------------------------------------------------------------------
class ModelB(Estimator):
    key = "B_massa_bulk"
    label = "B · Massa/Densidade a granel"
    color = "#ff7f0e"

    def compute(self, p):
        V = geometry.effective_volume_from_params(p)
        m_bean = beans.mass_from_params(p)
        return [Vi * rb / mb for Vi, rb, mb in zip(V, p["rho_bulk"], m_bean)]


# ---------------------------------------------------------------------------
# Modelo C — Estereologia / Visão computacional
# ---------------------------------------------------------------------------
class ModelC(Estimator):
    key = "C_estereologia"
    label = "C · Estereologia/Visão"
    color = "#2ca02c"

    def compute(self, p):
        V = geometry.effective_volume_from_params(p)
        out = []
        for Vi, na, t, c in zip(V, p["na_per_cm2"], p["t_layer_cm"], p["c_stereo"]):
            n_V = (na / t) * c          # densidade volumétrica de grãos (1/cm^3)
            out.append(n_V * Vi)
        return out


# ---------------------------------------------------------------------------
# Modelo D — Forma do grão + empacotamento por razão de aspecto
# ---------------------------------------------------------------------------
class ModelD(Estimator):
    key = "D_forma_mc"
    label = "D · Forma + Packing(aspecto)"
    color = "#9467bd"

    def compute(self, p):
        V = geometry.effective_volume_from_params(p)
        v_geo = beans.bean_volume_geometric(p["L_mm"], p["W_mm"], p["T_mm"], p["kappa"])
        aspect = beans.aspect_ratio(p["L_mm"], p["W_mm"], p["T_mm"])
        phi = packing.phi_from_shape(aspect, p["fric_factor"])
        return [Vi * ph / vg for Vi, ph, vg in zip(V, phi, v_geo)]


# ---------------------------------------------------------------------------
# Modelo G — Gravimétrico (PADRÃO-OURO, opcional: requer pesar o pote cheio)
# ---------------------------------------------------------------------------
class ModelGravimetric(Estimator):
    """N = (massa do pote cheio - massa do pote vazio) / massa por grão.

    Só é ativado se config.M_FULL_JAR_G estiver definido (não None). É de longe
    o método mais preciso: depende apenas da massa por grão (ancorada no
    experimento, com a pequena correção de tamanho s_lin) e de duas pesagens.
    """
    key = "G_gravimetrico"
    label = "G · Gravimétrico (pesagem)"
    color = "#8c564b"

    def compute(self, p):
        net = config.M_FULL_JAR_G - config.M_EMPTY_JAR_G
        return [net / (config.EXP_MASS_PER_BEAN_G * (s ** 3)) for s in p["s_lin"]]


BASE_MODELS = [ModelA(), ModelB(), ModelC(), ModelD()]


def active_models() -> List[Estimator]:
    """Modelos ativos nesta execução. Inclui o gravimétrico se houver pesagem
    do pote cheio (config.M_FULL_JAR_G)."""
    ms = [ModelA(), ModelB(), ModelC(), ModelD()]
    if config.M_FULL_JAR_G is not None:
        ms.append(ModelGravimetric())
    return ms

# Mapeamento modelo -> PRINCÍPIO de estimação independente.
# A e B são a MESMA física (rho_bulk ≈ phi·rho_app) em duas parametrizações:
# contam como UM princípio (senão haveria dupla contagem e falsa precisão no
# ensemble). C, D e G (gravimétrico) são princípios genuinamente independentes.
MODEL_TO_PRINCIPLE = {
    "A_volumetrico": "P_bulk_packing",
    "B_massa_bulk": "P_bulk_packing",
    "C_estereologia": "P_estereologia",
    "D_forma_mc": "P_forma_geom",
    "G_gravimetrico": "P_gravimetrico",
}
PRINCIPLE_LABEL = {
    "P_bulk_packing": "Bulk/Packing (A+B)",
    "P_estereologia": "Estereologia/Visão (C)",
    "P_forma_geom": "Forma+Packing (D)",
    "P_gravimetrico": "Gravimétrico (G)",
}


# ---------------------------------------------------------------------------
# Modelo E — Ensemble (pooling de PRINCÍPIOS independentes)
# ---------------------------------------------------------------------------
def build_principles(samples_by_model: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Agrega amostras por princípio (A e B viram um só)."""
    principles: Dict[str, List[float]] = {}
    for mkey, arr in samples_by_model.items():
        pkey = MODEL_TO_PRINCIPLE[mkey]
        principles.setdefault(pkey, []).extend(arr)
    return principles
def _consensus_weights(samples_by_group: Dict[str, List[float]]) -> Dict[str, float]:
    """Pesos por grupo (princípio) = precisão (1/CoV^2) x consistência.

    NÃO é um posterior Bayesiano (não há verossimilhança contra dados de N
    observado). É um POOLING PONDERADO transparente de princípios independentes:

    - Precisão: princípios mais precisos (menor CoV) pesam mais.
    - Consistência: penaliza um princípio cuja mediana se afasta do consenso,
      com escala = desvio-padrão típico dos princípios (régua física robusta),
      evitando o viés de espalhamento-de-medianas.

    Aplicado sobre PRINCÍPIOS (não modelos) para que A e B (redundantes) não
    contem como dois votos — eliminando falsa precisão.
    """
    keys = list(samples_by_group.keys())
    meds = {k: mcstats.median(samples_by_group[k]) for k in keys}
    sds = {k: mcstats.stdev(samples_by_group[k]) for k in keys}
    cvs = {k: max(mcstats.cov(samples_by_group[k]), 1e-3) for k in keys}

    grand_med = mcstats.median(list(meds.values()))
    scale = mcstats.median(list(sds.values()))
    if scale <= 0:
        scale = 0.15 * grand_med

    raw = {}
    for k in keys:
        precision = 1.0 / (cvs[k] ** 2)
        z = (meds[k] - grand_med) / scale
        consistency = pow(2.718281828, -0.5 * z * z)
        raw[k] = precision * consistency
    total = sum(raw.values())
    return {k: raw[k] / total for k in keys}


def linear_pool(samples_by_model: Dict[str, List[float]], weights: Dict[str, float], n: int, rng: RNG) -> List[float]:
    """Mistura ponderada: sorteia cada amostra de um modelo conforme os pesos."""
    keys = list(samples_by_model.keys())
    cum = []
    acc = 0.0
    for k in keys:
        acc += weights[k]
        cum.append(acc)
    pooled = []
    sizes = {k: len(samples_by_model[k]) for k in keys}
    for _ in range(n):
        u = rng.random()
        chosen = keys[-1]
        for k, c in zip(keys, cum):
            if u <= c:
                chosen = k
                break
        arr = samples_by_model[chosen]
        pooled.append(arr[int(rng.random() * sizes[chosen])])
    return pooled


def log_pool(samples_by_model: Dict[str, List[float]], weights: Dict[str, float], grid_n: int = 400):
    """Pool logarítmico: densidade proporcional ao PRODUTO ponderado das
    densidades (consenso "E"). Retorna (grade, densidade_normalizada).

    Mais concentrado que o linear quando os modelos concordam; pode quase
    zerar fora da região de concordância (por isso o reportamos como o
    estimador "de consenso", complementar ao linear)."""
    keys = list(samples_by_model.keys())
    all_vals = [v for k in keys for v in samples_by_model[k]]
    lo, hi = min(all_vals), max(all_vals)
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad
    grid = mcstats._linspace(lo, hi, grid_n)
    dx = grid[1] - grid[0]

    log_dens = [0.0] * grid_n
    for k in keys:
        dk = mcstats.kde_on_grid(samples_by_model[k], grid)
        wk = weights[k]
        for i, d in enumerate(dk):
            log_dens[i] += wk * (mcstats.math.log(d) if d > 0 else -50.0)
    mx = max(log_dens)
    dens = [pow(2.718281828, ld - mx) for ld in log_dens]
    z = sum(dens) * dx
    dens = [d / z for d in dens]
    return grid, dens


def grid_summary(grid: List[float], dens: List[float]) -> Dict[str, float]:
    """Resumo (média, mediana, IC95%) de uma densidade tabulada na grade."""
    dx = grid[1] - grid[0]
    # CDF
    cdf = []
    acc = 0.0
    for d in dens:
        acc += d * dx
        cdf.append(acc)
    total = cdf[-1]
    cdf = [c / total for c in cdf]

    def quant(q):
        for i in range(1, len(cdf)):
            if cdf[i] >= q:
                # interpolação linear
                t = (q - cdf[i - 1]) / (cdf[i] - cdf[i - 1] + 1e-12)
                return grid[i - 1] + t * (grid[i] - grid[i - 1])
        return grid[-1]

    mean = sum(g * d for g, d in zip(grid, dens)) * dx / total
    return {
        "mean": mean,
        "median": quant(0.5),
        "p2.5": quant(0.025),
        "p97.5": quant(0.975),
        "mode": grid[max(range(len(dens)), key=lambda i: dens[i])],
    }


def variance_decomposition(n: int, seed: int) -> Dict[str, float]:
    """Quanto da incerteza vem da GEOMETRIA do pote (V_eff), que é comum a
    todos os modelos? Calcula, no princípio Bulk/Packing (Modelo A), a fração
    da variância de N atribuível aos parâmetros geométricos.

    Estimador: Var condicional. Var_total (tudo varia) vs. a variância média
    de N quando a geometria é FIXADA (resto varia) -> a diferença, sobre o
    total, é a fração devida à geometria (índice de sensibilidade agrupado).
    """
    rng = RNG(seed)
    model = ModelA()
    specs = model.param_specs()
    geom = ["V_internal_cm3", "f_fill"]

    full = {nm: mcstats.sample_spec(specs[nm], n, rng.spawn(i)) for i, nm in enumerate(specs)}
    var_total = mcstats.variance(model.compute(full))

    # fixa geometria nos valores medianos; deixa o resto variar
    fixed = dict(full)
    for nm in geom:
        med = mcstats.median(full[nm])
        fixed[nm] = [med] * n
    var_no_geom = mcstats.variance(model.compute(fixed))

    frac_geom = max(0.0, min(1.0, 1.0 - var_no_geom / var_total)) if var_total > 0 else 0.0
    return {"frac_geometria": frac_geom, "frac_resto": 1.0 - frac_geom}


def run_all(n: int = N_SAMPLES, seed: int = 0) -> Dict:
    """Executa os modelos ativos + ensemble por princípios. Retorna resultados.

    Inclui o modelo gravimétrico (G) automaticamente se config.M_FULL_JAR_G
    estiver definido.
    """
    rng = RNG(seed)
    active = active_models()
    samples = {}
    for i, m in enumerate(active):
        samples[m.key] = m.sample(n, rng.spawn(1000 + i))

    principles = build_principles(samples)
    weights = _consensus_weights(principles)             # pesos por PRINCÍPIO
    pooled = linear_pool(principles, weights, n, rng.spawn(9999))
    grid, dens = log_pool(principles, weights)

    # peso por modelo (para tabelas): herda o peso do princípio
    model_weights = {mk: weights[MODEL_TO_PRINCIPLE[mk]] for mk in samples}

    return {
        "n": n,
        "models": {m.key: m for m in active},
        "samples": samples,
        "principles": principles,
        "weights": weights,                  # por princípio
        "model_weights": model_weights,      # por modelo (herdado)
        "ensemble_linear": pooled,
        "ensemble_log_grid": grid,
        "ensemble_log_dens": dens,
        "ensemble_log_summary": grid_summary(grid, dens),
        "variance_decomposition": variance_decomposition(min(n, 20000), seed + 1),
    }
