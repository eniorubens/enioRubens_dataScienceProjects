# Frequência de Sinistros por Apólice

[English version](README.en-US.md) · [Versão em português](README.pt-BR.md)

## 1. Resumo e resultado

Este projeto estuda a **frequência de sinistros** no benchmark francês de seguro automotivo `freMTPL2freq` (OpenML 41214, v1; 678.013 registros). A análise estima quantos sinistros são esperados por **100 apólices-ano** em diferentes perfis de veículo, condutor e região. Também mostra em quais grupos as previsões se aproximam dos totais observados e onde permanecem erros de calibração.

Foi feita uma única avaliação oficial no teste reservado, que reúne 101.859 registros, 53.839,09 apólices-ano e 5.415 sinistros. O **Poisson com offset de exposição fixo** (Cameron & Trivedi, 2013), congelado após o treino, alcançou desvio médio de Poisson de 0,31902, ante 0,33079 da taxa global de referência. A diferença pareada foi −0,01177 (IC 95% −0,01317 a −0,01018; por perfil, −0,01338 a −0,01028). Portanto, o teste sustenta o ganho do Poisson no desvio médio.

Na calibração agregada, foram observados 10,058 sinistros por 100 apólices-ano. O Poisson previu 10,108 e a taxa global, 10,084. O IC 95% da diferença prevista − observada inclui zero nos dois candidatos; no Poisson, vai de −0,233 a +0,317 e, na referência, de −0,253 a +0,288. Esses intervalos avaliam cada candidato separadamente; não testam equivalência entre eles nem comparam os erros absolutos pontuais, de 0,050 no Poisson e 0,027 na taxa global.

A principal limitação aparece na calibração por duração. Na faixa `(0; 0,02]`, a taxa observada foi 255,51, enquanto o Poisson previu 14,04 sinistros por 100 apólices-ano. As conclusões se restringem a este benchmark, e `production_eligible = false`.

## 2. Dados e pergunta

O `freMTPL2freq` está disponível no OpenML sob licença CC0. Cada registro representa a cobertura identificada por `IDpol`, com `ClaimNb` para a contagem de sinistros e `Exposure` para o tempo de exposição, em anos (Dutang & Charpentier, 2026). A avaliação verifica se o Poisson congelado supera a taxa global em registros que não participaram do desenvolvimento. Em seguida, identifica as faixas de exposição, idades do veículo e regiões nas quais as taxas previstas se afastam das observadas.

## 3. Estrutura do repositório

| Caminho | Conteúdo |
| --- | --- |
| `src/freq_sinistros/` | Lógica reutilizável (leitura, auditoria, partição, pré-processamento, modelos, métricas, congelamento, avaliação final, gráficos, rótulos PT/EN) |
| `notebooks/pt_br/`, `notebooks/en_us/` | Quatro notebooks por idioma, com o mesmo código: `01_eda_treino`, `02_comparacao_modelos`, `03_diagnostico_zeros`, `04_avaliacao_final` |
| `scripts/` | Geração e execução dos notebooks |
| `tests/` | Testes sintéticos (sem rede nem dados locais) |
| `outputs/etapa_01` … `etapa_05` | Resultados **agregados e versionáveis**, manifestos, hashes e registros de execução |
| `reports/` | Relatórios técnicos por etapa e protocolo de avaliação |
| `data/` | **Local, fora do versionamento**: o ARFF bruto (`data/raw`) e as atribuições de partição (`data/splits`) |

## 4. Mapa de leitura

O [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Claim_Frequency_per_Policy/) resume o problema, a decisão de modelagem e os principais resultados. Ele está publicado no GitHub Pages, e sua fonte, `site/index.html`, também pode ser aberta localmente no navegador. Para acompanhar a análise completa, os notebooks seguem a ordem abaixo.

| Notebook | Conteúdo |
| --- | --- |
| [01. EDA do treino](notebooks/pt_br/01_eda_treino.ipynb) | Apresenta as variáveis, o contrato de partição e a exploração feita somente no treino |
| [02. Comparação de modelos](notebooks/pt_br/02_comparacao_modelos.ipynb) | Explica a tabela analítica e compara taxa global, Poisson e NB2 na validação |
| [03. Diagnóstico de zeros e decisão](notebooks/pt_br/03_diagnostico_zeros.ipynb) | Avalia a necessidade de ZIP/ZINB, documenta a escolha do Poisson e verifica o congelamento |
| [04. Avaliação final](notebooks/pt_br/04_avaliacao_final.ipynb) | Concentra a conclusão no teste reservado; pode ser consultada sem o ARFF e sem reajustar os modelos, usando apenas os artefatos agregados publicados |

## 5. Ambiente

Ambiente conda (`environment.yml`: Python 3.12, numpy, pandas, scipy, statsmodels, matplotlib, nbclient, pyyaml, ruff, pytest). Observações desta máquina: `CONDA_SSL_VERIFY=truststore` para criar o ambiente, BLAS OpenBLAS (o MKL travava `numpy.linalg`) e `PYTHONNOUSERSITE=1` ao executar.

```powershell
$env:CONDA_SSL_VERIFY = "truststore"
conda env create -f environment.yml
conda activate freq-sinistros
$env:PYTHONNOUSERSITE = "1"
```

Para fixar as versões exatas usadas nas etapas 1 a 5 (Python 3.12.16, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2, OpenBLAS 0.3.34), há dois arquivos de lock. O `environment.lock.yml` exporta o ambiente (`conda env export --no-builds`) com versões exatas e sem builds, útil como referência e para outras plataformas, registrando os canais `rapidsai`, `defaults` e `conda-forge` herdados da configuração local da máquina, enquanto `environment.yml` usa só `conda-forge`. Já `conda-win-64.lock.txt` traz a lista explícita de pacotes para win-64, para recriar o ambiente exatamente com `conda create -n freq-sinistros --file conda-win-64.lock.txt`, seguido de `pip install -e . --no-deps`. Recomenda-se `environment.yml` para um ambiente novo e flexível, e o lock win-64 quando a reprodução byte a byte exigir as mesmas versões. O projeto exige Python 3.12 ou superior e é distribuído sob a licença MIT (`LICENSE`).

## 6. Consultar os resultados oficiais (sem dados locais)

A partir da raiz do projeto, o [notebook final](notebooks/pt_br/04_avaliacao_final.ipynb) pode ser executado com o ambiente da seção 5 e os artefatos já gravados em `outputs/`. Preserve os arquivos aprovados de `outputs/etapa_04` e `outputs/etapa_05`, incluindo `official_run.lock`.

```powershell
python -I -m freq_sinistros --root . verify-stage5     # confere offline os agregados oficiais
python scripts/run_notebooks.py --final               # executa só os dois notebooks finais, com os agregados gravados
```

O comando `verify-stage5` faz a conferência completa realizada pelo notebook `04_avaliacao_final`: verifica o registro (`status = completed`), os checksums de todos os arquivos listados em `SHA256SUMS_etapa05.txt` (e se não há arquivos fora da lista), a reconciliação, o modelo congelado e o roteiro contra os hashes do registro, o portão de integridade, o número de registros de teste (conferido com a etapa 2) e a presença da trava; se faltar o registro, a lista ou algum arquivo, termina com código 1 e uma mensagem, sem traceback. Os códigos de saída da CLI são 0 (sucesso), 1 (falha de verificação), 2 (auditoria divergente), 3 (aquisição recusada), 4 (`stage5` recusado) e 5 (`check-stage5` ou `verify-freeze` recusado após a trava). Quanto aos insumos locais dos demais notebooks, `01_eda_treino` lê o ARFF e as atribuições (`data/raw`, `data/splits`), e `02_comparacao_modelos` e `03_diagnostico_zeros` conferem o SHA-256 do ARFF; apenas o par `04` roda sem eles.

## 7. Reproduzir as etapas 1 a 4 (cópia isolada)

Use o ambiente conda da seção 5 e uma cópia fora do workspace. O download requer rede (≈ 36 MB). No ambiente registrado, o download levou cerca de 5 min e o fluxo levou cerca de 10 min. Nesta cópia são refeitas as etapas 1 a 3, e a etapa 4 é conferida pela restauração do modelo aprovado em validação. Para os notebooks finais, use a seção 6 com os agregados oficiais.

```powershell
$copy = Join-Path $env:TEMP ("freq-sinistros-" + [guid]::NewGuid())
robocopy . $copy /E /XD data .venv .pytest_tmp __pycache__ etapa_05
if ($LASTEXITCODE -ge 8) { throw "Robocopy failed." }
Set-Location $copy
python -I -m freq_sinistros --root . acquire           # baixa o ARFF (rede), confere o MD5 do OpenML
python -I -m freq_sinistros --root . audit             # etapa 1: auditoria
python -I -m freq_sinistros --root . partition         # etapa 2: partição 70/15/15 por IDpol
python -I -m freq_sinistros --root . eda               # etapa 3: EDA só com o treino
python -I -m freq_sinistros --root . compare           # etapa 3: taxa global, Poisson e NB2 em validação
python -I -m freq_sinistros --root . verify-freeze     # etapa 4: restaura o modelo congelado aprovado e reproduz a validação
python -I -m freq_sinistros --root . check-stage5      # portão de integridade (não disponibiliza o teste ao avaliador)
python -m pytest ; ruff check .
```

Em Linux ou macOS, o equivalente em bash (mesmos comandos Python):

```bash
copy="$(mktemp -d)/freq-sinistros"
mkdir -p "$copy"
rsync -a --exclude data --exclude .pytest_tmp --exclude __pycache__ --exclude outputs/etapa_05 ./ "$copy"/
cd "$copy"
python -I -m freq_sinistros --root . acquire
python -I -m freq_sinistros --root . audit
python -I -m freq_sinistros --root . partition
python -I -m freq_sinistros --root . eda
python -I -m freq_sinistros --root . compare
python -I -m freq_sinistros --root . verify-freeze
python -m pytest && ruff check .
```

A identidade byte a byte com os artefatos aprovados foi verificada no Windows: os JSON e CSV de `outputs/` foram gravados com fim de linha CRLF (a gravação usa o separador da plataforma). Em Linux ou macOS os arquivos regenerados terão LF; compare-os ignorando `\r` (por exemplo, `diff <(tr -d '\r' < a) b`). Por isso `check-stage5` não consta do bloco bash: o portão compararia os hashes dos seis arquivos da etapa 3 regenerados e os recusaria por diferença de fim de linha, não de conteúdo.

Os comandos `check-stage5` e `verify-freeze` só rodam na ausência da trava: na cópia isolada que exclui `etapa_05`, eles executam normalmente, ao passo que no workspace, onde existe `outputs/etapa_05/official_run.lock`, ambos recusam com código 5 e não gravam nada (pois regravariam artefatos selados). A cópia inclui `outputs/etapa_04` aprovado (`frozen_model.json` e `stage5_script.json`), portanto `stage4` não deve ser rodado sobre esses arquivos porque regeneraria o congelamento e os bytes do JSON podem mudar por incluírem informações de plataforma, levando o portão da etapa 5 a comparar os hashes com os valores aprovados e recusar a cópia regenerada. A verificação de integridade compara hashes byte a byte com os artefatos aprovados; em caso de divergência na reprodução, conferem-se as versões do ambiente contra `conda-win-64.lock.txt` e `environment.lock.yml` (seção 5). Da mesma forma, `stage5` nunca deve rodar no workspace porque a trava `outputs/etapa_05/official_run.lock` recusa uma segunda avaliação, razão pela qual a cópia omite `etapa_05` de propósito. O leitor interpreta o ARFF integral em memória, e os dados de teste só ficam acessíveis ao avaliador da etapa 5 após o portão de integridade e a trava.

## 8. Etapas

| Etapa | Síntese |
| --- | --- |
| 1. Aquisição e auditoria | ARFF verificado por MD5 e SHA-256; 0 faltantes; `IDpol` único |
| 2. Desenho da avaliação | Partição determinística por hash de `IDpol` (474.581 / 101.573 / 101.859); protocolo de métricas, segmentos, bootstrap e regras de decisão |
| 3. EDA e comparação | Taxa global, Poisson e NB2 em validação; sensibilidades |
| 4. Zeros, decisão e congelamento | Poisson com offset fixo escolhido pela regra aprovada; sem ZIP/ZINB; modelo congelado e verificado |
| 5. Avaliação final | Roteiro congelado aplicado uma vez ao teste (`stage5`) |

## 9. Limitações

O offset fixo **subestima as exposições muito curtas** e superestima as longas, sem recalibração aplicada. Os resultados restringem-se ao benchmark, sem evidência temporal, sem `BonusMalus` (cujo instante de medição é desconhecido) e sem modelos inflacionados de zeros, com intervalos de 95% condicionados aos modelos ajustados.

## 10. Verificação e CI

Os checks locais (Ruff, pytest com cobertura e paridade dos quatro pares de notebooks e dos READMEs) passaram, e a execução com dados reais está concluída e registrada em `outputs/` e `reports/`. As listas `SHA256SUMS*` de todas as etapas conferem e são testadas por `tests/test_outputs_integrity.py`, enquanto os registros de execução dos notebooks ficam em `outputs/notebook_execution_etapa03.json`, `outputs/notebook_execution_etapa04.json` e `outputs/notebook_execution_etapa05.json`, fora das pastas seladas, com a integridade oficial conferida por `verify-stage5`. O arquivo selado `SHA256SUMS_etapa05.txt` foi gravado no Windows com fim de linha CRLF e não foi regravado (em GNU `sha256sum -c`, converte-se antes com `tr -d '\r' < SHA256SUMS_etapa05.txt | sha256sum -c`; a CLI e o teste aceitam os dois formatos).

Para integração contínua, o projeto é publicado como a pasta `Claim_Frequency_per_Policy` do repositório de portfólio [enioRubens_dataScienceProjects](https://github.com/eniorubens/enioRubens_dataScienceProjects), no qual o workflow raiz `Tests` executa o `pytest` desta pasta como suíte própria da matriz (Python 3.12, `pip install -e ".[dev]"`, registrada em `.github/ci/test-projects.json` e conferida por `tests/test_ci_config.py`) apenas quando esta pasta muda ou na bateria completa, sem misturar dependências com os demais projetos; o CI usa `pip`, como as demais suítes do portfólio, enquanto o ambiente local permanece conda (seção 5) e o Ruff é checado localmente. O código tem licença MIT (`LICENSE`), e os dados do OpenML são CC0 e permanecem fora do repositório.

## 11. Referências

Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9781139013567

Dutang, C., & Charpentier, A. (2026). *CASdatasets: Insurance datasets* (R package version 1.2-1). https://doi.org/10.57745/P0KHAG

As referências completas de cada notebook, inclusive as que tratam dos zeros, estão ao final dele.
