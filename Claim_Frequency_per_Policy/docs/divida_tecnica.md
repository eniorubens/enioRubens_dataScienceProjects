# Dívida técnica

Data: 2026-11-23. Este registro reúne os achados A12 e A13 da auditoria (`supervisao/auditoria_projeto.md`) e os itens identificados durante a escrita dos testes da CLI, de `artifacts` e de `plots`.

## Como ler este documento

Os módulos abaixo têm o SHA-256 registrado no modelo congelado (`outputs/etapa_04/frozen_model.json`), no portão de integridade ou na execução oficial (`outputs/etapa_05/`). Qualquer alteração, mesmo de formatação ou de assinatura, muda o hash, reprova o portão (`stage5.integrity_gate`) e invalida a rastreabilidade da avaliação única no teste. **Por isso nenhum item da seção 1 foi corrigido.**

Esses itens só podem ser tratados em uma **nova versão do modelo**, com novo congelamento (etapa 4), novo conjunto reservado de dados e nova avaliação (etapa 5). O teste atual já foi consumido (Dwork et al., 2015); qualquer reuso terá de ser declarado como exploratório.

Os números de linha referem-se aos arquivos na data acima.

Módulos com hash localizado em `outputs/` (varredura de `json`, `csv` e `txt`): `arff`, `audit`, `compare`, `data`, `freeze`, `metrics`, `models`, `partition`, `prep`, `profiles`, `protocol`, `segments`, `stage4`, `stage5`. Nenhum outro módulo de `src/freq_sinistros/` tem o hash registrado.

## 1. Itens em módulos congelados

### 1.1 Argumento não usado: `models._update_alpha(start)`

- **Onde:** `src/freq_sinistros/models.py:166` (definição); chamadas em `models.py:249` (`start=None`) e `models.py:258` (`start=alpha`).
- **Efeito:** o parâmetro `start` nunca é lido. A maximização de `log(alpha)` usa `minimize_scalar(method="bounded")` sobre `[log ALPHA_MIN, log ALPHA_MAX]`, que não aceita ponto de partida. Embora a chamada passe o `alpha` anterior, não ocorre partida a quente. O resultado numérico não é afetado, pois a busca é limitada e determinística; o problema é a assinatura sugerir um comportamento que não existe.
- **Por que não foi corrigido:** `models.py` tem hash em `frozen_model.json` e em `official_run.json`.
- **Correção sugerida:** remover o parâmetro e as duas passagens de argumento. Se se quiser partida a quente de fato, estreitar os limites da busca em torno do `alpha` anterior e comparar a log-verossimilhança com a busca completa.

### 1.2 Argumento não usado: `freeze.code_hashes(root)`

- **Onde:** `src/freq_sinistros/freeze.py:35`; chamadas em `stage4.py:549` e `stage4.py:621`.
- **Efeito:** `root` é ignorado; os módulos são lidos do pacote efetivamente importado (`Path(__file__).resolve().parent`). A docstring já registra o fato ("mantido só por compatibilidade"). Quem passa uma raiz diferente acredita estar conferindo outro conjunto de módulos e não está.
- **Por que não foi corrigido:** `freeze.py` tem hash em `integrity_gate.json` e `official_run.json`; `stage4.py`, que chama a função, também.
- **Correção sugerida:** remover o parâmetro (e atualizar as duas chamadas) ou usá-lo de fato, como `package_dir` (o portão já tem `stage5.default_package_dir()` e `module_hashes(package_dir, ...)`, que fazem o equivalente).

### 1.3 Argumento não usado: `compare.point_signs(metrics_df)`

- **Onde:** `src/freq_sinistros/compare.py:357`; chamada em `compare.py:469` (`point_signs(metric_tables[-1], iv)`).
- **Efeito:** só `intervals` é usado; `metrics_df` é lido e descartado. O nome ("sinais") também não descreve o retorno, que são as estimativas pontuais, não os sinais.
- **Por que não foi corrigido:** `compare.py` tem hash em `official_run.json`.
- **Correção sugerida:** remover `metrics_df` da assinatura e da chamada; renomear para `point_estimates` ou devolver de fato `np.sign`.

### 1.4 Argumento não usado: `compare._evaluate_degree(val)`

- **Onde:** `src/freq_sinistros/compare.py:371`; chamada em `compare.py:450`.
- **Efeito:** o corpo ajusta e descreve o degrau usando somente `train`; `val` não é lido. A avaliação em validação ocorre depois, em `run_comparison`. Não há vazamento, mas a assinatura sugere que o ajuste de cada degrau enxerga a validação. Em uma auditoria, essa ambiguidade torna o fluxo mais difícil de verificar.
- **Por que não foi corrigido:** `compare.py` tem hash registrado.
- **Correção sugerida:** remover `val` da assinatura e da chamada.

### 1.5 Erros-padrão ingênuos em `models._standard_errors`

- **Onde:** `src/freq_sinistros/models.py:182-186`; usados em `models.py:217` (Poisson) e `models.py:277` (NB2); gravados como `*_naive_se` em `outputs/etapa_03/coefficients_n*.csv` (`compare.py:397`).
- **Efeito:** `sqrt(diag((X'WX)^-1))` assume a variância do modelo (Poisson: `Var = mu`; NB2 com `alpha` fixo no ponto estimado). Não há estimador sanduíche, nem incerteza de `alpha` no NB2. Com sobredispersão ou má especificação do offset, os erros-padrão são subestimados. Também usa `pinv`, o que mascara matrizes quase singulares sem aviso. Os coeficientes e as métricas de previsão não dependem disso; só a interpretação inferencial dos coeficientes.
- **Por que não foi corrigido:** `models.py` e `compare.py` têm hash registrado; os CSV de coeficientes têm hash em `frozen_model.json`.
- **Correção sugerida:** acrescentar erros-padrão robustos (sanduíche HC, ou por agrupamento de perfil, coerente com o bootstrap agrupado) e reportar a incerteza de `alpha`. Manter a coluna ingênua para comparação. Avisar quando `pinv` precisar truncar valores singulares.

### 1.6 `stage5.write_checksums` grava CRLF no Windows

- **Onde:** `src/freq_sinistros/stage5.py:471-472` (`Path.write_text("\n".join(lines) + "\n", encoding="ascii")`, sem `newline="\n"`).
- **Efeito:** no Windows, `\n` vira `\r\n`. `outputs/etapa_05/SHA256SUMS_etapa05.txt` fica com CRLF e `sha256sum -c` do GNU falha sem conversão (achado A9). `artifacts.verify_stage5` e `stage5.verify_official` leem com `splitlines()` e não notam. Em Linux o mesmo código geraria LF, de modo que a lista não é idêntica entre plataformas.
- **Por que não foi corrigido:** `stage5.py` tem hash em `official_run.json`; a lista oficial está selada e não deve ser regravada.
- **Correção sugerida:** usar `open(..., "w", encoding="ascii", newline="\n")` (ou `write_bytes`). Enquanto isso, documentar a conversão (`tr -d '\r'`) para conferir a lista da etapa 5 com GNU.

### 1.7 `stage5.verify_official` lança exceção em vez de devolver 1

- **Onde:** `src/freq_sinistros/stage5.py:496-524` (leitura sem proteção de `official_run.json` em `:509` e de `SHA256SUMS_etapa05.txt` em `:511`; `file_sha(out / n)` em `:517`).
- **Efeito:** a ausência do registro, da lista ou de um arquivo listado produz `FileNotFoundError`; `json.JSONDecodeError` e `ValueError` (linha da lista sem ` *`) também escapam. Assim, o contrato da docstring ("0 ... 1 caso contrário") não é cumprido. A função também é menos rigorosa que `artifacts.verify_stage5`: não confere a trava, os hashes do modelo e do roteiro contra o registro, nem o portão (achado A5).
- **Contorno atual:** a CLI (`cli.verify_stage5_full`) captura essas exceções e devolve 1, e combina as duas verificações. Há testes em `tests/test_cli.py`.
- **Por que não foi corrigido:** `stage5.py` tem hash registrado.
- **Correção sugerida:** tratar as exceções dentro de `verify_official` e unificar com `artifacts.verify_stage5` num único critério.

### 1.8 `integrity_gate` e `verify_freeze` gravam em pastas seladas (contornado na CLI)

- **Onde:** `src/freq_sinistros/stage5.py:183-187` (`integrity_gate.json` em `outputs/etapa_05/`, gravado **antes** de levantar `IntegrityError` em `:188-190`); `src/freq_sinistros/stage4.py:630` (`verify_freeze.json` em `outputs/etapa_04/`), chamado por `integrity_gate` em `stage5.py:173`.
- **Efeito:** os dois arquivos constam de `SHA256SUMS_etapa05.txt` e `SHA256SUMS_etapa04.txt`. Uma execução depois da avaliação oficial regrava os arquivos selados; se a checagem falhar (ex.: módulo editado), grava `ok: false` sobre o registro oficial e quebra `verify-stage5`. Foi o que aconteceu com os timestamps de 21/11 (achado A2).
- **Contorno atual (resolvido na CLI):** `cli.main` recusa `check-stage5` **e** `verify-freeze` com código 5 quando `outputs/etapa_05/official_run.lock` existe (função auxiliar `cli._locked_refusal`), sem gravar nada. A verificação do congelamento continua disponível numa cópia isolada sem `outputs/etapa_05`, como no README. Testes em `tests/test_cli.py` (`test_check_stage5_refuses_after_lock`, `test_verify_freeze_refuses_after_lock`, `test_verify_freeze_runs_when_not_locked`). A causa, nas funções congeladas, permanece.
- **Por que não foi corrigido:** `stage4.py` e `stage5.py` têm hash registrado.
- **Correção sugerida:** as funções receberem o destino de saída (`out_dir`) ou um parâmetro `write=False`, e gravar por padrão numa pasta não selada (ex.: `outputs/checks/`); na CLI, aplicar a recusa por trava também a `verify-freeze`.

### 1.9 Fim de linha dependente da plataforma na gravação de JSON/CSV

Item encontrado ao inspecionar `outputs/`.

- **Onde:** `src/freq_sinistros/stage3.py:24` (`write_json`, usada por `stage4`, `stage5` e `compare`), `design.py:121`, `audit.py:716` e `:806`, `acquire.py:171` e `:223`. Todas usam `Path.write_text(...)` sem `newline="\n"`; os CSV gravados por `pandas.to_csv(path, ..., lineterminator="\n")` (`stage3.write_table`, `audit.py:722`, `design.py:221-231`) também saíram com CRLF no Windows, porque o pandas abre o arquivo em modo texto e o `\n` é traduzido para o separador da plataforma.
- **Efeito:** no Windows, os JSON e CSV de `outputs/` têm CRLF (ex.: `frozen_model.json`: 462 linhas, todas CRLF; medido: todos os 83 arquivos de `outputs/` têm CRLF); em Linux/macOS teriam LF. Os hashes aprovados (`stage5.APPROVED["frozen_model_sha256"]` e outros) valem para os bytes com CRLF. Em Linux/macOS o portão da etapa 5 (que confere os 6 arquivos da etapa 3) reprovaria uma reprodução idêntica em conteúdo. O CI atual só roda testes sintéticos, de modo que a divergência não aparece.
- **Por que não foi corrigido agora:** `stage3.py`, `design.py` e `acquire.py` **não** têm hash registrado, mas mudar o fim de linha alteraria os bytes reproduzidos no Windows e, portanto, os hashes aprovados; `audit.py`, `stage4.py` e `stage5.py` têm hash. A correção parcial deixaria a cadeia inconsistente.
- **Correção sugerida:** numa nova versão, gravar com `newline="\n"` (JSON, via `Path.open`) e `lineterminator="\n"` com o arquivo aberto em `newline=""` (CSV) em todos os módulos, e congelar de novo, registrando os novos hashes.
- **Contorno atual:** documentado nos READMEs (seção 7): a identidade byte a byte vale para o Windows; em Linux/macOS compara-se ignorando `\r`, e `check-stage5` foi retirado do bloco bash. `.gitattributes` (`outputs/** -text`) preserva os bytes CRLF no repositório.

### 1.10 Fim de linha misto entre módulos

- **Onde:** `src/freq_sinistros/`. Medição atual: `arff.py`, `partition.py`, `profiles.py`, `protocol.py`, `__init__.py`, `__main__.py` e `notebook_checks.py` têm só LF; os demais têm só CRLF. Nenhum arquivo individual é misto.
- **Efeito:** o SHA-256 depende do fim de linha. Um checkout com conversão automática (`core.autocrlf`) mudaria os hashes dos módulos registrados e reprovaria o portão.
- **Contorno atual:** `.gitattributes` marca os 14 módulos com hash como `-text !eol`, preservando os bytes; os demais seguem `*.py text eol=lf`. Ao normalizar os módulos não congelados para LF no primeiro commit, o hash deles (não registrado) muda sem consequência.
- **Por que não foi corrigido:** normalizar os módulos congelados mudaria o hash.
- **Correção sugerida:** numa nova versão, normalizar todos para LF antes do novo congelamento, e remover as exceções do `.gitattributes`.

## 2. Itens encontrados ao escrever os testes (módulos não congelados)

Estes itens não exigem novo congelamento; foram só registrados, sem alteração de comportamento.

### 2.1 `cli.main`: `verify-freeze` sem proteção contra a trava (resolvido)

Resolvido: ver 1.8. `verify-freeze` recusa com código 5 quando `outputs/etapa_05/official_run.lock` existe.

### 2.2 `artifacts.provenance_table`: código morto

- **Onde:** `src/freq_sinistros/artifacts.py` (`provenance_table`).
- **Efeito:** monta uma lista `values` de 7 itens (com `names[6]` e `""`) e depois usa só `values[:6]`/`names[:6]`; o sétimo rótulo (`"registro de cobertura (IDpol)"`) existe nos dois idiomas, mas nunca aparece. Sem erro, apenas ruído.
- **Correção sugerida:** reduzir a lista a seis itens ou exibir o sétimo.

### 2.3 `artifacts.project_root` e o diretório base dos testes

- **Efeito:** `pyproject.toml` define `--basetemp=.pytest_tmp`, dentro do projeto. Assim, `tmp_path` fica abaixo de uma pasta com `pyproject.toml` e o erro de `project_root` não pode ser provocado com `tmp_path`. O teste em `tests/test_artifacts_views.py` simula a ausência trocando `Path.exists` com `monkeypatch`.
- **Correção sugerida:** só se `--basetemp` mudar (para fora do projeto); então o teste pode dispensar o `monkeypatch`.

### 2.4 `cli.main` com complexidade 12 e `print` direto

- **Onde:** `src/freq_sinistros/cli.py:78` (regras `C901`/`T201` do ruff, fora das regras do projeto). A cadeia de `if` por comando poderia ser uma tabela de despacho (`dict[str, Callable]`). Não há risco funcional; o ganho é legibilidade, e os testes de `tests/test_cli.py` cobrem cada ramo.

### 2.5 Tipagem estática

- As funções públicas de `cli.py`, `artifacts.py`, `plots.py`, `labels.py`, `notebook_checks.py`, `acquire.py`, `design.py`, `eda.py` e `stage3.py` estão anotadas (`ruff --select ANN001,ANN201` sem apontamentos). Ficam sem anotação as funções privadas `acquire._open` e `design._load_verified` (retorno), e os módulos congelados (cerca de 100 argumentos).
- Não há verificação estática (`mypy`/`pyright`). Sugestão: adicioná-la ao CI para os módulos não congelados e estender aos congelados só na nova versão.
