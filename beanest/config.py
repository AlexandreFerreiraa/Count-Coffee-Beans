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
ESPECIFICAÇÕES MEDIDAS DO POTE (fornecidas pelo usuário) -- DADO DURO
----------------------------------------------------------------------------
- Material: vidro transparente.
- Capacidade interna (cheio até a borda): 3223 mL = 3223 cm^3.  <== usar isto!
- Altura total: 24.5 cm. Diâmetro EXTERNO (corpo): 14.7 cm. Tampa (metálica): 113 mm.
- Lacre termoencolhível transparente. Massa do pote VAZIO: 1240 g.

ATENÇÃO — DIÂMETRO EXTERNO vs INTERNO:
  Os 14.7 cm são o diâmetro EXTERNO. O diâmetro INTERNO é menor:
      D_interno = D_externo - 2 * espessura_do_vidro
  Para vidro deste porte a parede tem ~3-4 mm, logo D_interno ~ 13.9-14.1 cm.
  PORÉM: a contagem de grãos NÃO usa o diâmetro para calcular volume — usa a
  CAPACIDADE INTERNA MEDIDA (3223 cm^3), que já é o volume interno real. Logo a
  distinção externo/interno NÃO enviesa N. O diâmetro interno é usado apenas
  como (a) checagem de coerência geométrica e (b) referência da visão
  computacional (ver beanest.geometry.geometry_consistency e WALL_THICKNESS_CM).

CONSEQUÊNCIA METODOLÓGICA (melhoria importante):
  Antes estimávamos o volume por um cilindro a partir de diâmetro/altura
  (com ~60% da incerteza total vindo daí). Agora usamos a CAPACIDADE MEDIDA
  (3223 cm^3) e apenas a FRAÇÃO PREENCHIDA pelos grãos (f_fill) é incerta — o
  pescoço/headspace fica vazio. Isso reduz muito a incerteza geométrica.
      V_eff = V_internal * f_fill
  Conferência: um cilindro de 14.7 cm (externo) por 24.5 cm daria ~3700 cm^3;
  a capacidade real (3223) é menor por causa do afunilamento do pescoço E da
  parede de vidro — logo a capacidade medida é mais fiel que a aproximação
  cilíndrica baseada no diâmetro externo.

----------------------------------------------------------------------------
PROVENIÊNCIA DAS OBSERVAÇÕES (3 fotos) E DO GRÃO
----------------------------------------------------------------------------
- Grãos: TORRADOS (confirmado pelo usuário). Torra muda a física do grão:
  o grão incha (fica maior) e fica POROSO (mais leve por volume).
- Os grãos não chegam ao topo: há headspace vazio no pescoço (-> f_fill < 1,
  estimado das fotos).
- Experimento do usuário: 89 grãos pesam 11 g -> 0.12360 g/grão (referência de
  massa por grão; assumimos que são os mesmos grãos torrados do pote).
- O usuário observou grãos do pote ligeiramente maiores -> s_lin >= 1 (modesto).

----------------------------------------------------------------------------
VALORES DE LITERATURA -- CAFÉ TORRADO (whole bean)
----------------------------------------------------------------------------
- Massa por grão torrado: ~0.12-0.17 g (consistente com 11/89 = 0.1236 g).
- Densidade APARENTE (envelope) do grão torrado: ~0.55-0.80 g/cm^3 (poroso;
  bem menor que o verde ~1.1-1.2, pois a torra cria porosidade interna).
- Densidade A GRANEL (bulk) do grão torrado inteiro: ~0.33-0.45 g/cm^3
  (cerca de metade do café verde; café torrado é "leve e fofo").
- Fração de empacotamento phi ~ 0.55-0.62 (geometria do empacotamento depende
  pouco da torra; phi ~ rho_bulk / rho_app ~ 0.39/0.68 ~ 0.57).
- Empacotamento de esferoides (Donev et al., Science 2004): densidade de
  empacotamento aleatório sobe de ~0.64 (esfera) para ~0.71 perto de razão de
  aspecto ~1.3-1.5, decrescendo suavemente depois. Fator de atrito reduz para
  grãos reais. (Valores reimplementados/parafraseados para conformidade.)

Esses números entram como PRIORS com incerteza; os dados (pote medido +
experimento de massa + fotos) ancoram o modelo. Ajuste-os em config se tiver
medições melhores. A medição que mais reduz a incerteza é PESAR O POTE CHEIO
(ver M_FULL_JAR_G e o modelo gravimétrico).
"""

# Constante do experimento caseiro (dado duro fornecido pelo usuário)
EXP_BEANS = 89
EXP_MASS_G = 11.0
EXP_MASS_PER_BEAN_G = EXP_MASS_G / EXP_BEANS  # 0.123595... g/grão

# ---------------------------------------------------------------------------
# ESPECIFICAÇÕES MEDIDAS DO POTE / MODO GRAVIMÉTRICO (padrão-ouro opcional)
# ---------------------------------------------------------------------------
M_EMPTY_JAR_G = 1240.0   # massa do pote VAZIO (com tampa/lacre), fornecida

# Dimensões medidas do pote (cm). ATENÇÃO: o diâmetro é EXTERNO.
D_EXT_CM = 14.7          # diâmetro EXTERNO do corpo (medido)
JAR_HEIGHT_CM = 24.5     # altura total
LID_DIAM_CM = 11.3       # tampa metálica (113 mm)
V_INTERNAL_MEASURED_CM3 = 3223.0  # capacidade interna medida

# Espessura da parede de vidro (cm). Não entra na contagem (usamos o volume
# medido), mas serve para estimar o diâmetro INTERNO e checar coerência.
# Vidro deste porte: parede ~3-4 mm. Faixa para o cross-check de coerência:
WALL_THICKNESS_CM = 0.35
WALL_THICKNESS_LO_CM = 0.25
WALL_THICKNESS_HI_CM = 0.45

# >>> PREENCHA AQUI PARA O MELHOR RESULTADO POSSÍVEL <<<
# Se você pesar o pote CHEIO (grãos + pote) numa balança e colocar o valor (g)
# em M_FULL_JAR_G, o modelo gravimétrico é ativado:
#     N = (M_FULL_JAR_G - M_EMPTY_JAR_G) / massa_por_grão
# Ele é muito mais preciso que qualquer estimativa por volume e passa a dominar
# o ensemble. Deixe como None se não tiver a pesagem.
M_FULL_JAR_G = None      # ex.: 2350.0  (pote cheio em gramas)

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
    # ---- Geometria do pote: VOLUME MEDIDO * FRAÇÃO PREENCHIDA ----
    "V_internal_cm3": {
        "dist": "truncnormal", "mean": 3223.0, "sd": 60.0, "lo": 3050.0, "hi": 3400.0,
        "units": "cm^3", "desc": "Volume interno do pote (capacidade medida)",
        "source": "Especificação do usuário: capacidade 3223 mL.",
    },
    "f_fill": {
        "dist": "truncnormal", "mean": 0.89, "sd": 0.035, "lo": 0.78, "hi": 0.98,
        "units": "-", "desc": "Fração do volume interno ocupada pela coluna de grãos",
        "source": "Das fotos: grãos vão até ~o ombro; pescoço/headspace vazio.",
    },

    # ---- Grão TORRADO: tamanho/massa/densidade ----
    "s_lin": {
        "dist": "truncnormal", "mean": 1.03, "sd": 0.04, "lo": 0.95, "hi": 1.15,
        "units": "-", "desc": "Fator de escala LINEAR dos grãos do pote vs. experimento",
        "source": "Grãos do pote ~iguais/ligeiramente maiores que o experimento; massa ~ s_lin^3.",
    },
    "rho_app": {
        "dist": "truncnormal", "mean": 0.68, "sd": 0.06, "lo": 0.55, "hi": 0.82,
        "units": "g/cm^3", "desc": "Densidade aparente (envelope) do grão TORRADO (poroso)",
        "source": "Literatura café torrado ~0.55-0.80 g/cm^3.",
    },

    # ---- Empacotamento ----
    "phi": {
        "dist": "truncnormal", "mean": 0.57, "sd": 0.030, "lo": 0.49, "hi": 0.64,
        "units": "-", "desc": "Fração de empacotamento a granel (grãos irregulares alongados)",
        "source": "Random packing de grãos elipsoidais com atrito ~0.55-0.62.",
    },
    "rho_bulk": {
        "dist": "truncnormal", "mean": 0.39, "sd": 0.035, "lo": 0.32, "hi": 0.47,
        "units": "g/cm^3", "desc": "Densidade a granel (bulk) do café TORRADO inteiro",
        "source": "Literatura/indústria ~0.33-0.45 g/cm^3 (torrado é leve/fofo).",
    },

    # ---- Modelo C: estereologia / visão computacional ----
    # na_per_cm2 e t_layer_cm podem ser SOBRESCRITOS pela CV (ver vision.py).
    "na_per_cm2": {
        "dist": "truncnormal", "mean": 2.0, "sd": 0.5, "lo": 1.0, "hi": 3.5,
        "units": "1/cm^2", "desc": "Densidade areal de grãos visíveis na parede do vidro",
        "source": "Contagem visual nas fotos (grão torrado é maior -> menos por cm^2).",
    },
    "t_layer_cm": {
        "dist": "truncnormal", "mean": 0.70, "sd": 0.10, "lo": 0.45, "hi": 1.00,
        "units": "cm", "desc": "Espessura efetiva da 1ª camada (profundidade média do grão)",
        "source": "Profundidade média do grão torrado na direção de visão.",
    },
    "c_stereo": {
        "dist": "truncnormal", "mean": 1.00, "sd": 0.15, "lo": 0.70, "hi": 1.40,
        "units": "-", "desc": "Fator de calibração estereológica (efeito de parede etc.)",
        "source": "Empacotamento na parede difere do interior; absorve viés do método.",
    },

    # ---- Modelo D: forma do grão TORRADO (Monte Carlo geométrico) ----
    # Dimensões a partir de FAIXAS DE LITERATURA para grão torrado (inchado pela
    # torra). Rota INDEPENDENTE do experimento de massa: concordância = validação.
    "L_mm": {
        "dist": "truncnormal", "mean": 11.5, "sd": 1.3, "lo": 8.0, "hi": 15.5,
        "units": "mm", "desc": "Comprimento do grão torrado (eixo maior)",
        "source": "Faixa de literatura para café torrado; ancorável pela CV.",
    },
    "W_mm": {
        "dist": "truncnormal", "mean": 7.6, "sd": 0.9, "lo": 5.5, "hi": 10.5,
        "units": "mm", "desc": "Largura do grão torrado (eixo médio)",
        "source": "Faixa de literatura para café torrado; ancorável pela CV.",
    },
    "T_mm": {
        "dist": "truncnormal", "mean": 5.3, "sd": 0.8, "lo": 3.5, "hi": 7.5,
        "units": "mm", "desc": "Espessura do grão torrado (eixo menor; incha na torra)",
        "source": "Faixa de literatura para café torrado (~5 mm).",
    },
    "kappa": {
        "dist": "truncnormal", "mean": 0.71, "sd": 0.07, "lo": 0.55, "hi": 0.85,
        "units": "-", "desc": "Fator de forma: volume sólido do grão / volume do elipsoide envolvente",
        "source": "Grão torrado é mais 'cheio'/arredondado que o verde; fator < 1.",
    },
    "fric_factor": {
        "dist": "truncnormal", "mean": 0.86, "sd": 0.03, "lo": 0.78, "hi": 0.94,
        "units": "-", "desc": "Redução de phi por atrito/irregularidade vs. esferoide ideal",
        "source": "Grãos reais (atrito) empacotam mais frouxo que esferoides lisos.",
    },
}


# Quais parâmetros cada modelo consome (usado pela análise de Sobol).
MODEL_PARAMS = {
    "A_volumetrico": ["V_internal_cm3", "f_fill", "phi", "s_lin", "rho_app"],
    "B_massa_bulk": ["V_internal_cm3", "f_fill", "rho_bulk", "s_lin"],
    "C_estereologia": ["V_internal_cm3", "f_fill", "na_per_cm2", "t_layer_cm", "c_stereo"],
    "D_forma_mc": ["V_internal_cm3", "f_fill", "L_mm", "W_mm", "T_mm", "kappa", "fric_factor"],
    "G_gravimetrico": ["s_lin"],
}


def effective_volume_cm3(V_internal_cm3, f_fill):
    """Volume efetivo ocupado pelos grãos = volume interno medido * fração
    preenchida. Usar a capacidade medida (em vez de um cilindro idealizado)
    reduz drasticamente a incerteza geométrica; o headspace do pescoço entra
    apenas via f_fill < 1.
    """
    return V_internal_cm3 * f_fill
