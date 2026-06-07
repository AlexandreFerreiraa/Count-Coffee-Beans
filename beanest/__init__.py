"""
beanest — Estimação estatística do número de grãos de café num pote cilíndrico.

Pacote modular, núcleo sem dependências (stdlib apenas). Aceleradores opcionais
(matplotlib, scipy, opencv) são usados se estiverem instalados, com degradação
graciosa.

Módulos:
    config     parâmetros, priors e proveniência (fontes) de cada valor
    mcstats    utilitários de Monte Carlo e estatística (stdlib)
    geometry   volume interno efetivo da coluna de grãos
    beans      propriedades do grão (massa, densidade, volume, tamanho)
    packing    fração de empacotamento (random packing) e phi(razão de aspecto)
    models     estimadores A, B, C, D e o ensemble Bayesiano E
    compare    comparação de modelos, sensibilidade (Sobol) e convergência
    vision     pipeline de visão computacional (opcional)
    plotting   gráficos SVG (puro Python) + matplotlib opcional
    report     montagem do relatório (Markdown + HTML)
"""

__version__ = "1.0.0"
