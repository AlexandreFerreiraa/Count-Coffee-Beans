"""
Configuração central: parâmetros incertos (priors), constantes e proveniência.

Cada parâmetro é declarado como um *spec* (dicionário) com:
    dist   : "const" | "normal" | "truncnormal" | "lognormal"
    ...     : parâmetros da distribuição (mean, sd, lo, hi, value)
    units  : unidade física
    source : justificativa/fonte do valor (observação das imagens, experimento,
             literatura). Documentar a proveniência é parte da metodologia.
    desc   : descrição curta

A filosofia: TODA quantidade que não conhecemos com exatidão é tratada como
variável aleatória. A propagação de incerteza é feita por Monte Carlo
(ver beanest.mcstats e beanest.models).

----------------------------------------------------------------------------
PROVENIÊNCIA DAS OBSERVAÇÕES (extraídas das 3 fotos fornecidas pelo usuário)
----------------------------------------------------------------------------
- O pote é um frasco cilíndrico de vidro com tampa, segurado por uma mão
  masculina adulta (referência de escala: largura da palma ~8.5 cm, comprimento
  punho->ponta do dedo médio ~19 cm).
- Pela proporção mão/pote, o frasco se assemelha a um "garrafão"/jar de ~1 galão
  (~3.8 L) a 4 L: diâmetro externo ~14-15 cm, altura do corpo ~24-27 cm, com
  pescoço afunilando para uma tampa de ~10-11 cm.
- Os grãos NÃO chegam ao topo: há espaço vazio no pescoço (o usuário confirma).
  Por isso modelamos diretamente a ALTURA DA COLUNA DE GRÃOS (h_beans), e não a
  altura total do pote — isso já contabiliza o vazio do pescoço de forma direta.
- Grãos: aparentam ser café VERDE (cru) especial; alguns grãos claros/brancos
  parecem ligeiramente maiores que os mais escuros (o usuário aponta isso).
  -> tratamos os grãos do pote como ~6% maiores (linearmente) que os do
     experimento caseiro, com incerteza (parâmetro s_lin).
- Experimento do usuário: 89 grãos pesam 11 g  ->  0.12360 g/grão (referência).

----------------------------------------------------------------------------
PROVENIÊNCIA DOS VALORES DE LITERATURA (café verde/cru)
----------------------------------------------------------------------------
- Massa por grão de café verde especial: ~0.12-0.18 g (consistente com o
  experimento 11/89 = 0.1236 g).
- Densidade aparente (do sólido do grão, "envelope") do café verde: ~1.05-1.25
  g/cm^3 (grão verde tem baixa porosidade; densidade real do sólido ~1.3-1.4).
- Densidade a granel ("bulk") do café verde: ~0.60-0.74 g/cm^3
  (regra de ofício do setor; varia com a variedade e umidade).
- Fração de empacotamento de grãos irregulares/alongados (random packing):
  phi ~ 0.55-0.65 (esferas monodispersas: random close packing ~0.64; grãos
  elipsoidais alongados com atrito tendem a empacotar um pouco mais frouxo).
- Empacotamento de esferoides (Donev et al., Science 2004): a densidade máxima
  de empacotamento aleatório sobe de ~0.64 (esfera) para ~0.71 perto de razão
  de aspecto ~1.3-1.5, decrescendo para razões maiores. Aplicamos um fator de
  atrito/irregularidade para grãos reais. (Valores reimplementados/parafraseados
  para conformidade.)

Esses números entram como PRIORS com incerteza; os dados (experimento + fotos)
ancoram o modelo. Ajuste-os em config se tiver medições melhores.
"""

from math import pi

# Constante do experimento caseiro (dado duro fornecido pelo usuário)
EXP_BEANS = 89
EXP_MASS_G = 11.0
EXP_MASS_PER_BEAN_G = EXP_MASS_G / EXP_BEANS  # 0.123595... g/grão

# Semente para reprodutibilidade
RANDOM_SEED = 20240607

# Tamanho amostral de Monte Carlo (núcleo). 60k dá distribuições suaves e roda
# em segundos em Python puro. Aumente para suavizar ainda mais as caudas.
N_SAMPLES = 60_000

# Tamanho amostral para a análise de sensibilidade de Sobol (estimador de
# Saltelli; custo = N_SOBOL * (k + 2) avaliações por modelo).
N_SOBOL = 4_000


# ---------------------------------------------------------------------------
# REGISTRO DE PARÂMETROS
# ---------------------------------------------------------------------------
# Truncamos as normais em limites físicos plausíveis para evitar valores
# absurdos nas caudas (ex.: densidade negativa).

PARAMS = {
    # ---- Geometria do pote (volume efetivo da coluna de grãos) ----
    "d_in_cm": {
        "dist": "truncnormal", "mean": 14.0, "sd": 1.0, "lo": 11.0, "hi": 17.0,
        "units": "cm", "desc": "Diâmetro interno do corpo do pote",
        "source": "Estimado das fotos pela escala da mão (palma ~8.5 cm).",
    },
    "h_beans_cm": {
        "dist": "truncnormal", "mean": 22.5, "sd": 2.0, "lo": 16.0, "hi": 28.0,
        "units": "cm", "desc": "Altura da COLUNA de grãos (já exclui o vazio do pescoço)",
        "source": "Estimado das fotos; grãos não alcançam o topo.",
    },
    "k_shape": {
        "dist": "truncnormal", "mean": 0.97, "sd": 0.02, "lo": 0.90, "hi": 1.00,
        "units": "-", "desc": "Correção de forma (fundo arredondado/irregularidade da seção)",
        "source": "Cilindro idealizado superestima ligeiramente o volume útil.",
    },

    # ---- Grão: tamanho/massa/densidade ----
    "s_lin": {
        "dist": "truncnormal", "mean": 1.06, "sd": 0.04, "lo": 0.95, "hi": 1.20,
        "units": "-", "desc": "Fator de escala LINEAR dos grãos do pote vs. experimento",
        "source": "Usuário observa grãos do pote ligeiramente maiores; massa ~ s_lin^3.",
    },
    "rho_app": {
        "dist": "truncnormal", "mean": 1.13, "sd": 0.07, "lo": 0.95, "hi": 1.30,
        "units": "g/cm^3", "desc": "Densidade aparente (envelope) do sólido do grão verde",
        "source": "Literatura café verde ~1.05-1.25 g/cm^3.",
    },

    # ---- Empacotamento ----
    "phi": {
        "dist": "truncnormal", "mean": 0.585, "sd": 0.030, "lo": 0.50, "hi": 0.66,
        "units": "-", "desc": "Fração de empacotamento a granel (grãos irregulares alongados)",
        "source": "Random packing de grãos elipsoidais com atrito ~0.55-0.65.",
    },
    "rho_bulk": {
        "dist": "truncnormal", "mean": 0.66, "sd": 0.04, "lo": 0.55, "hi": 0.74,
        "units": "g/cm^3", "desc": "Densidade a granel (bulk) do café verde",
        "source": "Literatura/indústria ~0.60-0.74 g/cm^3.",
    },

    # ---- Modelo C: estereologia / visão computacional ----
    # na_per_cm2 e t_layer_cm podem ser SOBRESCRITOS pela CV (ver vision.py).
    "na_per_cm2": {
        "dist": "truncnormal", "mean": 2.6, "sd": 0.6, "lo": 1.2, "hi": 4.5,
        "units": "1/cm^2", "desc": "Densidade areal de grãos visíveis na parede do vidro",
        "source": "Contagem visual aproximada nas fotos (1ª camada na parede).",
    },
    "t_layer_cm": {
        "dist": "truncnormal", "mean": 0.62, "sd": 0.10, "lo": 0.40, "hi": 0.90,
        "units": "cm", "desc": "Espessura efetiva da 1ª camada (profundidade média do grão)",
        "source": "Profundidade média do grão na direção de visão.",
    },
    "c_stereo": {
        "dist": "truncnormal", "mean": 1.00, "sd": 0.15, "lo": 0.70, "hi": 1.40,
        "units": "-", "desc": "Fator de calibração estereológica (efeito de parede etc.)",
        "source": "Empacotamento na parede difere do interior; absorve viés do método.",
    },

    # ---- Modelo D: forma do grão (Monte Carlo geométrico) ----
    # Dimensões do grão de café verde a partir de FAIXAS DE LITERATURA
    # (independentes do experimento de massa). O Modelo D é, portanto, uma rota
    # de estimação independente: se concordar com A/B, isso é uma validação
    # genuína; se divergir, é informação sobre rho_app/kappa. NÃO ajustamos estes
    # valores para casar com a rota de massa.
    "L_mm": {
        "dist": "truncnormal", "mean": 11.0, "sd": 1.3, "lo": 7.5, "hi": 15.0,
        "units": "mm", "desc": "Comprimento do grão (eixo maior)",
        "source": "Faixa de literatura para café verde; ancorável pela CV.",
    },
    "W_mm": {
        "dist": "truncnormal", "mean": 7.1, "sd": 0.9, "lo": 5.0, "hi": 10.0,
        "units": "mm", "desc": "Largura do grão (eixo médio)",
        "source": "Faixa de literatura para café verde; ancorável pela CV.",
    },
    "T_mm": {
        "dist": "truncnormal", "mean": 4.4, "sd": 0.7, "lo": 3.0, "hi": 6.3,
        "units": "mm", "desc": "Espessura do grão (eixo menor)",
        "source": "Faixa de literatura para café verde (grão chato ~4-5 mm).",
    },
    "kappa": {
        "dist": "truncnormal", "mean": 0.65, "sd": 0.07, "lo": 0.50, "hi": 0.82,
        "units": "-", "desc": "Fator de forma: volume sólido do grão / volume do elipsoide envolvente",
        "source": "Grão de café ~ meio-elipsoide 'cheio' (face plana); fator < 1 (incerteza ampla).",
    },
    "fric_factor": {
        "dist": "truncnormal", "mean": 0.86, "sd": 0.03, "lo": 0.78, "hi": 0.94,
        "units": "-", "desc": "Redução de phi por atrito/irregularidade vs. esferoide ideal",
        "source": "Grãos reais (atrito) empacotam mais frouxo que esferoides lisos.",
    },
}


# Quais parâmetros cada modelo consome (usado pela análise de Sobol).
MODEL_PARAMS = {
    "A_volumetrico": ["d_in_cm", "h_beans_cm", "k_shape", "phi", "s_lin", "rho_app"],
    "B_massa_bulk": ["d_in_cm", "h_beans_cm", "k_shape", "rho_bulk", "s_lin"],
    "C_estereologia": ["d_in_cm", "h_beans_cm", "k_shape", "na_per_cm2", "t_layer_cm", "c_stereo"],
    "D_forma_mc": ["d_in_cm", "h_beans_cm", "k_shape", "L_mm", "W_mm", "T_mm", "kappa", "fric_factor"],
}


def effective_volume_cm3(d_in_cm, h_beans_cm, k_shape):
    """Volume da coluna de grãos = cilindro (corpo) * correção de forma.

    Modelar h_beans (altura dos grãos) em vez da altura total embute
    automaticamente o vazio do pescoço.
    """
    return pi * (d_in_cm / 2.0) ** 2 * h_beans_cm * k_shape
