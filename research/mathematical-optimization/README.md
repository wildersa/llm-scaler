# Mathematical Kernel Optimization Research

## Objetivo

Investigar formulações matemáticas alternativas para operadores do OmniXPU, buscando reduzir operações, tráfego de memória e temporários sem comprometer a semântica ou a estabilidade numérica.

## Referência principal

[Arquitetura e plano de otimização do OmniXPU](../../PLAN_BMG_VIDEO_DIT_KERNEL_OPTIMIZATION.md). A revisão de referência do plano é o commit `30c648d93998e9d798f7056e8ed0868b68a605e8`. Cada experimento deve identificar também a revisão exata do operador estudado.

## Método

1. Formular hipótese falsificável e identificar a formulação original.
2. Demonstrar equivalência matemática ou delimitar o erro e as condições de validade.
3. Executar experimentos reproduzíveis com referências independentes em CPU e casos sintéticos e representativos, incluindo casos extremos.
4. Registrar resultados, inclusive negativos, precisão, complexidade, operações, bytes e limitações.
5. Replicar os resultados promissores de forma independente.
6. Validar o desempenho em XPU e no workload real antes de promover qualquer alteração ao fork.

A redução de custo analítico não comprova ganho de desempenho na GPU. A pesquisa não substitui o inventário, baseline e profiling previstos no plano mestre.

## Fronteira com o fork

Esta pasta registra o escopo, referências e decisões de integração da pesquisa. Experimentos exploratórios, notebooks, dependências e dados de benchmark devem permanecer em um laboratório independente, preferencialmente em repositório próprio, para preservar o alinhamento do fork com o upstream. O laboratório independente está em [wildersa/kernel-research-lab](https://github.com/wildersa/kernel-research-lab), com o [protocolo de pesquisa](https://github.com/wildersa/kernel-research-lab/blob/main/research/RESEARCH_BRIEF.md).

Somente resultados validados e candidatos pertinentes ao OmniXPU devem avançar para o Kernel Workbench e para alterações versionadas dos operadores, respeitando contratos, retrocompatibilidade e os critérios de aceitação do plano mestre.

Nenhuma hipótese ou ganho experimental é considerado confirmado por este documento.
