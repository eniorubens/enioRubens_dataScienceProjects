# Handoff — Análise de viabilidade do projeto C-MAPSS

## Objetivo

Construir um projeto de portfólio sobre manutenção preditiva industrial usando o conjunto NASA C-MAPSS — Turbofan Engine Degradation Simulation. O escopo previsto inclui previsão de Remaining Useful Life (RUL), validação por motor, early warning, custo de falso alarme versus falha não detectada, explicabilidade e intervalos de incerteza.

## Localização

- Projeto: `D:\Projetos\CMAPSS_Predictive_Maintenance`
- ZIP original: `data\raw\CMAPSSData.zip`
- Dados extraídos: `data\raw\dataset`
- Fonte oficial: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

## Inspeção inicial

- ZIP: aproximadamente 11,85 MB.
- Dados descompactados: aproximadamente 50 MB.
- Treino: 160.359 linhas.
- Teste: 104.823 linhas.
- Total: aproximadamente 265 mil observações.
- Cada linha possui 26 colunas: motor, ciclo, 3 configurações operacionais e sensores.
- `train_FD001`: 20.631 linhas, 100 motores, ciclo máximo 362.
- `train_FD002`: 53.759 linhas, 260 motores, ciclo máximo 378.
- `train_FD003`: 24.720 linhas, 100 motores, ciclo máximo 525.
- `train_FD004`: 61.249 linhas; a contagem local encontrou 249 identificadores, enquanto o catálogo NASA informa 248; ciclo máximo 543.
- Os arquivos `RUL_FD001.txt` a `RUL_FD004.txt` estão presentes.

## Conclusão preliminar

O projeto é computacionalmente pequeno e não requer Spark, GPU ou cloud. O principal risco é metodológico: há poucos motores independentes, forte dependência temporal e risco de vazamento caso a divisão seja feita por linhas em vez de trajetórias.

## Próximas etapas

1. Auditar estruturalmente os arquivos e investigar a discrepância de FD004.
2. Explorar ciclos por motor, sensores constantes/redundantes, valores ausentes, escalas e comportamento temporal.
3. Começar com FD001 para um baseline reproduzível.
4. Validar por motor/trajetória, nunca por amostragem aleatória de linhas.
5. Expandir para FD002–FD004 como teste de generalização para múltiplas condições e modos de falha.
6. Comparar baseline simples, modelo de RUL, early-warning e análise de custo.

## Limites do que foi verificado

Até aqui foi feita apenas inspeção estrutural local de arquivos, linhas, motores e ciclos. Ainda não foi feita análise estatística dos sensores, verificação detalhada de valores ausentes, modelagem, validação preditiva ou avaliação de desempenho.

Não instalar plugins neste momento: a análise local pode ser feita com Python/PowerShell e scikit-learn. Plugins só seriam úteis para publicação, armazenamento externo ou versionamento remoto.
