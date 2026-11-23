# Frequência de Sinistros por Apólice

[English version](README.en-US.md) · [Versão em português](README.pt-BR.md)

## 1. Resumo e resultado

Estudo reproduzível de **frequência de sinistros** no benchmark francês de seguro automotivo `freMTPL2freq` (OpenML 41214, v1; 678.013 registros). O objetivo é estimar quantos sinistros esperar por **100 apólices-ano** em perfis de veículo, condutor e região e medir onde o modelo aproxima bem os totais observados e onde os subestima ou superestima.

**Resultado da avaliação final** (teste reservado, 101.859 registros, 53.839,09 apólices-ano, 5.415 sinistros; uma única execução): o **Poisson com offset de exposição fixo**, congelado no treino, tem desvio médio de Poisson 0,31902, contra 0,33079 da taxa global de referência. A diferença pareada é −0,01177 (IC 95% −0,01317 a −0,01018; por perfil, −0,01338 a −0,01028), então o teste **sustenta um ganho no desvio médio** sobre a referência. Na calibração total, a taxa observada é 10,058 por 100 apólices-ano e as previstas são 10,108 (Poisson) e 10,084 (taxa global). Cada IC 95% se refere à diferença prevista − observada de um candidato (Poisson −0,233 a +0,317; referência −0,253 a +0,288) e ambos incluem zero; esses intervalos individuais não testam equivalência entre os candidatos nem comparam seus erros absolutos, que são, em valores pontuais, 0,050 (Poisson) e 0,027 (taxa global). O desvio por duração persiste: na faixa `(0; 0,02]`, o observado é 255,51 e o previsto 14,04. As conclusões valem dentro do benchmark; `production_eligible = false`.

## 2. Dados e pergunta

`freMTPL2freq`, licença CC0 no OpenML. A unidade é o registro de cobertura (`IDpol`) com `ClaimNb` (contagem) e `Exposure` (anos). A pergunta final: o Poisson congelado supera a taxa global em registros nunca usados, e onde as taxas previstas se afastam das observadas por faixa de exposição, idade do veículo e região?

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

O [relatório executivo](site/index.html) apresenta o problema, a decisão e os principais resultados em uma leitura breve. Abra o arquivo localmente no navegador: no GitHub um `.html` aparece como código-fonte (publicar via GitHub Pages é opcional). Os notebooks preservam a narrativa analítica completa:

1. [EDA do treino](notebooks/pt_br/01_eda_treino.ipynb) — apresenta as variáveis, o contrato de partição e a exploração feita somente no treino.
2. [Comparação de modelos](notebooks/pt_br/02_comparacao_modelos.ipynb) — explica a tabela analítica e compara taxa global, Poisson e NB2 na validação.
3. [Diagnóstico de zeros e decisão](notebooks/pt_br/03_diagnostico_zeros.ipynb) — avalia a necessidade de ZIP/ZINB, documenta a escolha do Poisson e verifica o congelamento.
4. **[Avaliação final](notebooks/pt_br/04_avaliacao_final.ipynb)** — concentra a conclusão no teste reservado. Pode ser consultada sem o ARFF e sem reajustar os modelos, usando apenas os artefatos agregados publicados.

## 5. Ambiente

Ambiente conda (`environment.yml`: Python 3.12, numpy, pandas, scipy, statsmodels, matplotlib, nbclient, pyyaml, ruff, pytest). Observações desta máquina: `CONDA_SSL_VERIFY=truststore` para criar o ambiente, BLAS OpenBLAS (o MKL travava `numpy.linalg`) e `PYTHONNOUSERSITE=1` ao executar.

```powershell
$env:CONDA_SSL_VERIFY = "truststore"
conda env create -f environment.yml
conda activate freq-sinistros
$env:PYTHONNOUSERSITE = "1"
```

Para fixar as versões exatas usadas nas etapas 1 a 5 (Python 3.12.16, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2, OpenBLAS 0.3.34), há dois arquivos de lock:

- `environment.lock.yml`: exportação do ambiente (`conda env export --no-builds`) com versões exatas e sem builds, útil como referência e para outras plataformas. Atenção: ele registra os canais `rapidsai`, `defaults` e `conda-forge`, herdados da configuração local da máquina, enquanto `environment.yml` usa só `conda-forge`.
- `conda-win-64.lock.txt`: lista explícita de pacotes para win-64, para recriar o ambiente exatamente: `conda create -n freq-sinistros --file conda-win-64.lock.txt`, seguido de `pip install -e . --no-deps`.

Use `environment.yml` para um ambiente novo e flexível; use o lock win-64 quando a reprodução byte a byte precisar das mesmas versões. O projeto exige Python 3.12 ou superior e é distribuído sob a licença MIT (`LICENSE`).

## 6. Consultar os resultados oficiais (sem dados locais)

A partir da raiz do projeto, o [notebook final](notebooks/pt_br/04_avaliacao_final.ipynb) pode ser executado com o ambiente da seção 5 e os artefatos já gravados em `outputs/`. Preserve os arquivos aprovados de `outputs/etapa_04` e `outputs/etapa_05`, incluindo `official_run.lock`.

```powershell
python -I -m freq_sinistros --root . verify-stage5     # confere offline os agregados oficiais
python scripts/run_notebooks.py --final               # executa só os dois notebooks finais, com os agregados gravados
```

- `verify-stage5` faz a conferência completa, a mesma do notebook `04_avaliacao_final`: o registro (`status = completed`), os checksums de todos os arquivos listados em `SHA256SUMS_etapa05.txt` (e se não há arquivos fora da lista), a reconciliação, o modelo congelado e o roteiro contra os hashes do registro, o portão de integridade, o número de registros de teste (conferido com a etapa 2) e a presença da trava. Se faltar o registro, a lista ou um arquivo, ele termina com código 1 e uma mensagem, sem traceback.
- Códigos de saída da CLI: 0 sucesso; 1 falha de verificação; 2 auditoria divergente; 3 aquisição recusada; 4 `stage5` recusado; 5 `check-stage5` ou `verify-freeze` recusado após a trava.
- Insumos locais dos demais notebooks: `01_eda_treino` lê o ARFF e as atribuições (`data/raw`, `data/splits`); `02_comparacao_modelos` e `03_diagnostico_zeros` conferem o SHA-256 do ARFF. Só o par `04` roda sem eles.

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

- `check-stage5` e `verify-freeze` só rodam onde não há trava: a cópia exclui `etapa_05` e por isso eles rodam normalmente. No workspace, com `outputs/etapa_05/official_run.lock`, os dois recusam com código 5 e não gravam nada (regravariam artefatos selados).
- A cópia leva `outputs/etapa_04` aprovada (`frozen_model.json`, `stage5_script.json`). **Não** rode `stage4` sobre esses arquivos: ele regenera o congelamento, e os bytes do JSON podem mudar porque incluem informações de plataforma; o portão da etapa 5 compara os hashes com os valores aprovados e recusaria a cópia regenerada.
- A verificação de integridade compara hashes byte a byte com os artefatos aprovados. Se houver divergência na reprodução, confira as versões do ambiente contra `conda-win-64.lock.txt` e `environment.lock.yml` (seção 5).
- Nunca rode `stage5` no workspace: a trava `outputs/etapa_05/official_run.lock` recusa uma segunda avaliação. A cópia exclui `etapa_05` de propósito.
- O leitor interpreta o ARFF integral em memória; o teste só é disponibilizado ao avaliador da etapa 5 depois do portão de integridade e da trava.

## 8. Etapas

1. **Aquisição e auditoria**: ARFF verificado por MD5 e SHA-256; 0 faltantes; `IDpol` único.
2. **Desenho da avaliação**: partição determinística por hash de `IDpol` (474.581 / 101.573 / 101.859); protocolo de métricas, segmentos, bootstrap e regras de decisão.
3. **EDA e comparação**: taxa global, Poisson e NB2 em validação; sensibilidades.
4. **Zeros, decisão e congelamento**: Poisson com offset fixo escolhido pela regra aprovada; sem ZIP/ZINB; modelo congelado e verificado.
5. **Avaliação final**: roteiro congelado aplicado uma vez ao teste (`stage5`).

## 9. Limitações

- O offset fixo **subestima as exposições muito curtas** e superestima as longas; não houve recalibração.
- Resultados internos ao benchmark: sem evidência temporal, sem `BonusMalus` (instante de medição desconhecido) e sem modelos inflacionados de zeros.
- Os intervalos de 95% são condicionados aos modelos ajustados.

## 10. Verificação e CI

- **Checks locais** (Ruff, pytest com cobertura, paridade dos quatro pares de notebooks e dos READMEs): passaram.
- **Execução com dados reais**: concluída e registrada em `outputs/` e `reports/`.
- **Listas de checksum**: as listas `SHA256SUMS*` de todas as etapas conferem e são testadas por `tests/test_outputs_integrity.py`. Os registros de execução dos notebooks ficam em `outputs/notebook_execution_etapa03.json`, `outputs/notebook_execution_etapa04.json` e `outputs/notebook_execution_etapa05.json`, fora das pastas seladas. A integridade da avaliação oficial é conferida por `verify-stage5`.
- **Fim de linha da etapa 5**: `SHA256SUMS_etapa05.txt` está com CRLF (gravado no Windows; é um arquivo selado e não foi regravado). Para o `sha256sum -c` do GNU, converta antes, por exemplo `tr -d '\r' < SHA256SUMS_etapa05.txt | sha256sum -c`. A CLI e o teste aceitam os dois formatos.
- **CI**: o projeto é publicado como a pasta `Claim_Frequency_per_Policy` do repositório de portfólio [enioRubens_dataScienceProjects](https://github.com/eniorubens/enioRubens_dataScienceProjects). O workflow raiz `Tests` roda o `pytest` desta pasta como suíte própria da matriz (Python 3.12, `pip install -e ".[dev]"`), registrada em `.github/ci/test-projects.json`, só quando esta pasta muda (ou na bateria completa), sem misturar dependências com os demais projetos. `tests/test_ci_config.py` confere esse registro. O CI usa `pip`, como as demais suítes do portfólio; o ambiente local continua conda (seção 5). O Ruff é verificado localmente.
- **Licença**: MIT (`LICENSE`), para o código; os dados do OpenML são CC0 e ficam fora do repositório.

## 11. Referências

Dutang e Charpentier (2018), *CASdatasets: Insurance datasets* (dicionário de `freMTPL2freq`); Cameron e Trivedi (2013), *Regression Analysis of Count Data*; Wilson (2015), *Economics Letters*, 127, 51–53. Referências completas ao final de cada notebook.
