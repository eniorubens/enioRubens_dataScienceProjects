"""Estimadores de CATE (S/T/X/R-learner, Causal Forest, Uplift RF) sob interface comum.

Populado incrementalmente: baseline de propensão e T-learner provisório em S3,
meta-learners completos em S4, Causal Forest e Uplift Trees em S5.
"""
import hashlib
import importlib.metadata as importlib_metadata

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from causalml.inference.meta import BaseRRegressor, BaseSRegressor, BaseTRegressor, BaseXRegressor
from causalml.inference.tree import UpliftRandomForestClassifier, UpliftTreeClassifier
from econml.dml import CausalForestDML
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder

from .config import BIN_VARS, CAT_VARS, CONT_VARS, FEATURE_COLS, SEED

META_LEARNER_NAMES = ("S", "T", "X", "R")


def _prepare_features(df):
    """Recorta as colunas de feature e marca as categóricas para o LightGBM.

    O LightGBM detecta automaticamente colunas pandas de dtype `category` —
    evita expandir `history_segment`/`zip_code`/`channel` em one-hot.
    """
    X = df[FEATURE_COLS].copy()
    for col in CAT_VARS:
        X[col] = X[col].astype("category")
    return X


def fit_propensity_baseline(train_df, treatment_col, outcome_col, seed=SEED):
    """Treina P(outcome=1 | X) usando só as linhas do braço tratado.

    É o baseline "o que marketing faz na prática": um modelo de resposta
    treinado sobre quem recebeu o tratamento, sem nenhuma noção de efeito
    incremental. Usa LightGBM — mesma classe de modelo dos meta-learners de
    S4 — para que a diferença de desempenho isole a estratégia de targeting
    (propensão vs. uplift), não a capacidade do modelo.
    """
    treated = train_df[train_df[treatment_col] == 1]
    X = _prepare_features(treated)
    y = treated[outcome_col]
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    model.fit(X, y)
    return model


def predict_propensity_score(model, df):
    """P(outcome=1 | X) prevista pelo baseline, para qualquer conjunto de linhas."""
    return model.predict_proba(_prepare_features(df))[:, 1]


def fit_t_learner_quick(train_df, treatment_col, outcome_col, seed=SEED):
    """T-learner rápido (2 LightGBM independentes) como referência provisória de uplift em S3.

    Não tunado e não comparado formalmente aos demais learners — isso é
    trabalho de S4. Serve aqui só para checar se o ranking por propensão
    concorda com um ranking por efeito incremental.
    """
    treated = train_df[train_df[treatment_col] == 1]
    control = train_df[train_df[treatment_col] == 0]

    model_treated = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    model_treated.fit(_prepare_features(treated), treated[outcome_col])

    model_control = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    model_control.fit(_prepare_features(control), control[outcome_col])

    return model_treated, model_control


def predict_t_learner_uplift(models, df):
    """CATE estimado pelo T-learner: P(Y=1|X,T=1) - P(Y=1|X,T=0)."""
    model_treated, model_control = models
    X = _prepare_features(df)
    return model_treated.predict_proba(X)[:, 1] - model_control.predict_proba(X)[:, 1]


def build_meta_learner_encoder(train_df, cat_vars=None):
    """Ajusta (só no treino) um OneHotEncoder para as categóricas usadas pelos meta-learners.

    O `causalml` converte `X` para numpy puro internamente (`X.to_numpy()`
    dentro de `fit`/`predict`), o que descarta o dtype `category` do pandas de
    que o LightGBM depende para lidar nativamente com categóricas — a mesma
    técnica usada em `fit_propensity_baseline`/`fit_t_learner_quick` não
    sobrevive a essa conversão. Por isso, aqui as categóricas são
    one-hot-encodadas antes de entrar no meta-learner.

    `cat_vars=None` usa `CAT_VARS` (as 3 categóricas originais de S4.1-4.4).
    S4.5 passa `features.EXTENDED_CAT_VARS` para incluir as interações
    derivadas sem tocar o comportamento default.
    """
    cat_vars = cat_vars if cat_vars is not None else CAT_VARS
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[cat_vars])
    return encoder


def encode_meta_learner_features(df, encoder, cont_vars=None, bin_vars=None, cat_vars=None):
    """Matriz numérica (contínuas + binárias + categóricas em one-hot) para os meta-learners.

    `cont_vars`/`bin_vars`/`cat_vars=None` usam os globais de `config.py`
    (comportamento default, idêntico a S4.1-4.4). Passe as listas
    `EXTENDED_*_VARS` de `features.py` para a versão com features derivadas
    de S4.5 — `encoder` precisa ter sido ajustado com o mesmo `cat_vars`.
    """
    cont_vars = cont_vars if cont_vars is not None else CONT_VARS
    bin_vars = bin_vars if bin_vars is not None else BIN_VARS
    cat_vars = cat_vars if cat_vars is not None else CAT_VARS
    numeric = df[cont_vars + bin_vars].to_numpy(dtype=float)
    cat_encoded = encoder.transform(df[cat_vars])
    return np.hstack([numeric, cat_encoded])


def _default_base_learner(seed=SEED):
    return lgb.LGBMRegressor(random_state=seed, verbose=-1)


def fit_single_meta_learner(name, X, treatment, y, seed=SEED, base_learner_factory=None, p=None):
    """Ajusta um único meta-learner (S/T/X/R) sobre features/tratamento/outcome já preparados.

    `base_learner_factory` é um callable sem argumentos que devolve uma nova
    instância do regressor de base a cada chamada — default `None` usa o
    LightGBM vanilla de sempre (`_default_base_learner`). Usado tanto pelo
    ajuste completo dos quatro learners (`fit_meta_learners`) quanto por
    diagnósticos que só precisam de um learner por vez (ex.: chão de ruído
    por permutação em S4.4, sensibilidade ao algoritmo base em S4.6).

    `p`, se fornecido (array alinhado a `X`/`treatment`), é repassado direto
    ao `fit` do `causalml` — usado pela ablation de propensão conhecida em
    S4.7. **Efeito mecânico não é igual entre learners** (verificado via
    inspeção do código-fonte do causalml 0.15.5): no X-learner, `p` no fit
    não afeta os modelos mu/tau (são ajustados igual independente de `p`) —
    seu único papel aqui é impedir que `_set_propensity_models` rode e sete
    `self.propensity_model`, o que muda o comportamento do `predict`
    (ver `predict_single_meta_learner`). No R-learner, `p` entra diretamente
    na perda R (`(y - yhat) / (w - p)`, peso `(w - p)**2`) — afeta de fato o
    que `models_tau` aprende. Em S/T, `p` é aceito na assinatura por
    consistência de interface, mas nunca usado.
    """
    factory = base_learner_factory or (lambda: _default_base_learner(seed))
    builders = {
        "S": lambda: BaseSRegressor(learner=factory()),
        "T": lambda: BaseTRegressor(learner=factory()),
        "X": lambda: BaseXRegressor(learner=factory()),
        "R": lambda: BaseRRegressor(learner=factory(), random_state=seed),
    }
    model = builders[name]()
    model.fit(X, treatment, y, p=p)
    return model


def predict_single_meta_learner(name, model, X, p=None):
    """Predição de um único meta-learner, repassando `p` quando fornecido.

    Só o **X-learner** usa `p` de fato no predict (pondera a combinação final
    `p*tau_c + (1-p)*tau_t`) — para ele, `p` aqui precisa ser o vetor
    alinhado ao `X` sendo predito agora (não o mesmo array usado no fit).
    R-learner aceita `p` na assinatura mas o ignora; S/T não usam `p`.
    Se `name == 'X'` e o modelo foi ajustado com `p` explícito no fit, chamar
    `predict` sem `p` aqui levanta `TypeError` ('NoneType' object is not
    subscriptable — verificado empiricamente: `self.propensity_model` existe
    como atributo `None` por padrão, não é um atributo ausente, então o erro
    real não é `AttributeError`). Passar o mesmo tipo de `p` em ambas as
    chamadas evita esse problema.
    """
    if name == "R":
        return model.predict(X).ravel()
    return model.predict(X, p=p).ravel()


def fit_meta_learners(train_df, treatment_col, outcome_col, encoder, seed=SEED, base_learner_factory=None):
    """Ajusta os quatro meta-learners (S/T/X/R) com um único base learner comum.

    Outcome binário tratado via variantes *Regressor* (regressão da
    probabilidade) nos quatro — não *Classifier* — porque o R-learner do
    `causalml` só existe como Regressor; usar Classifier nos outros três
    quebraria a comparação ao misturar capacidades de modelo diferentes entre
    learners. Propensão de tratamento (usada por X- e R-learner) é estimada
    automaticamente pelo `causalml` (`p=None`), de forma consistente entre
    fit e predict. `base_learner_factory=None` usa LightGBM vanilla (S4.1-4.3);
    ver `fit_single_meta_learner` para variar o algoritmo base (S4.6).
    """
    X = encode_meta_learner_features(train_df, encoder)
    treatment = train_df[treatment_col].to_numpy()
    y = train_df[outcome_col].to_numpy(dtype=float)
    return {
        name: fit_single_meta_learner(name, X, treatment, y, seed=seed, base_learner_factory=base_learner_factory)
        for name in META_LEARNER_NAMES
    }


def predict_meta_learners_uplift(models, df, encoder, cont_vars=None, bin_vars=None, cat_vars=None):
    """CATE de cada meta-learner (dict {nome: array}) para qualquer conjunto de linhas.

    Passe as mesmas listas `cont_vars`/`bin_vars`/`cat_vars` usadas para
    ajustar `encoder` (relevante só para a iteração de S4.5, com features
    estendidas — o default `None` cobre S4.1-4.4 sem mudança).
    """
    X = encode_meta_learner_features(df, encoder, cont_vars=cont_vars, bin_vars=bin_vars, cat_vars=cat_vars)
    return {name: model.predict(X).ravel() for name, model in models.items()}


def _regularized_base_learner(seed=SEED):
    """Base learner mais raso/conservador para X/R (S4.5): num_leaves=15 (default
    31), max_depth=5 (default -1), min_child_samples=50 (default 20),
    reg_alpha=reg_lambda=1.0 (default 0). `n_estimators`/`learning_rate`
    ficam intocados — isola o eixo de complexidade da árvore.

    Racional (Künzel et al., 2019 p/ X-learner; Nie & Wager, 2021 p/
    R-learner): ambos constroem o alvo do estágio final a partir de modelos
    de incômodo ruidosos no primeiro estágio — uma árvore final mais rasa e
    conservadora ataca diretamente essa amplificação de ruído.
    """
    return lgb.LGBMRegressor(
        random_state=seed, verbose=-1,
        num_leaves=15, max_depth=5, min_child_samples=50,
        reg_alpha=1.0, reg_lambda=1.0,
    )


def fit_meta_learners_regularized(train_df, treatment_col, outcome_col, encoder, seed=SEED,
                                    cont_vars=None, bin_vars=None, cat_vars=None,
                                    learner_names=("X", "R"), base_learner_factory=None):
    """Ajusta um subconjunto de meta-learners (X/R por padrão — S e T ficam como
    referência inalterada de S4.1-4.4) para a iteração de S4.5.

    `base_learner_factory=None` usa `_regularized_base_learner` (eixo de
    regularização, configs B/C de S4.5); passe explicitamente
    `lambda: _default_base_learner(seed)` para isolar o efeito só das
    features, com hiperparâmetros vanilla (config A de S4.5).
    `cont_vars`/`bin_vars`/`cat_vars` aceitam `features.EXTENDED_*_VARS` para
    incluir as interações derivadas (`encoder` precisa ter sido ajustado com
    o mesmo `cat_vars`).
    """
    factory = base_learner_factory or (lambda: _regularized_base_learner(seed))
    X = encode_meta_learner_features(train_df, encoder, cont_vars=cont_vars, bin_vars=bin_vars, cat_vars=cat_vars)
    treatment = train_df[treatment_col].to_numpy()
    y = train_df[outcome_col].to_numpy(dtype=float)
    return {
        name: fit_single_meta_learner(name, X, treatment, y, seed=seed, base_learner_factory=factory)
        for name in learner_names
    }


def _package_versions():
    """Versões das libs cujo comportamento afeta o fit dos meta-learners —
    usadas no fingerprint do cache de `get_meta_learners` (S4-cont, item 5)."""
    return {
        "causalml": importlib_metadata.version("causalml"),
        "lightgbm": importlib_metadata.version("lightgbm"),
        "scikit-learn": importlib_metadata.version("scikit-learn"),
    }


def _meta_learners_cache_metadata(train_df, treatment_col, outcome_col, seed):
    """Metadata para invalidar o cache de `get_meta_learners`: fingerprint
    SHA-256 dos dados relevantes ao fit (features + tratamento + outcome),
    lista de features, seed, hiperparâmetros concretos do base learner e
    versões de pacote. `get_meta_learners` só reusa um artefato salvo em
    disco se esta metadata bater exatamente com a configuração atual — evita
    a fragilidade anterior, em que `RETRAIN=False` reusava silenciosamente um
    artefato desatualizado após qualquer mudança nos dados, features ou
    versão de biblioteca.

    `base_learner_params` (via `.get_params()`) existe além do nome
    `base_learner` porque o nome sozinho não pega uma mudança futura interna
    a `_default_base_learner` (ex.: um hiperparâmetro ajustado mantendo o
    mesmo nome) — sem isso, um cache antigo poderia parecer válido mesmo após
    o base learner de fato mudar.
    """
    relevant = train_df[FEATURE_COLS + [treatment_col, outcome_col]]
    data_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(relevant, index=True).values.tobytes()
    ).hexdigest()
    return {
        "data_fingerprint": data_fingerprint,
        "feature_cols": list(FEATURE_COLS),
        "seed": seed,
        "base_learner": "LightGBM vanilla (_default_base_learner)",
        "base_learner_params": _default_base_learner(seed).get_params(),
        "package_versions": _package_versions(),
    }


def save_meta_learners(models, encoder, path, metadata=None):
    """Serializa os quatro meta-learners e o encoder juntos, em um único arquivo.

    `metadata`, se fornecida, é gravada junto — usada por `get_meta_learners`
    para detectar cache desatualizado (ver `_meta_learners_cache_metadata`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"models": models, "encoder": encoder}
    if metadata is not None:
        bundle["metadata"] = metadata
    joblib.dump(bundle, path)


def load_meta_learners(path):
    """Carrega os meta-learners e o encoder salvos por `save_meta_learners`."""
    bundle = joblib.load(path)
    return bundle["models"], bundle["encoder"]


def get_meta_learners(train_df, treatment_col, outcome_col, retrain, path, seed=SEED):
    """Treina (e salva) ou carrega os meta-learners, seguindo a flag `RETRAIN` do projeto.

    Se `retrain` for False e `path` já existir, tenta carregar do disco — mas
    só reusa o artefato se sua metadata (fingerprint dos dados, features,
    seed, versões de pacote — `_meta_learners_cache_metadata`) bater
    exatamente com a configuração atual; caso contrário, imprime um aviso e
    retreina, em vez de devolver silenciosamente um modelo desatualizado. Na
    primeira execução (artefato ainda não existe), treina e salva mesmo com
    `retrain=False`, para não travar a inicialização do projeto.
    """
    current_metadata = _meta_learners_cache_metadata(train_df, treatment_col, outcome_col, seed)
    if not retrain and path.exists():
        bundle = joblib.load(path)
        if bundle.get("metadata") == current_metadata:
            return bundle["models"], bundle["encoder"]
        print(
            f"Cache de meta-learners em {path} está desatualizado em relação à "
            "configuração atual (dados/features/seed/versões de pacote mudaram) "
            "— retreinando."
        )
    encoder = build_meta_learner_encoder(train_df)
    models = fit_meta_learners(train_df, treatment_col, outcome_col, encoder, seed=seed)
    save_meta_learners(models, encoder, path, metadata=current_metadata)
    return models, encoder


def fit_s_learner_linear_interaction(X, treatment, y, alpha=1.0, l1_ratio=0.5, seed=SEED):
    """S-learner linear com interações T×X_j explícitas (S4.10, item opcional).

    **Por que não usar `BaseSRegressor` (causalml) para isto.** Verificado via
    `inspect.getsource`: `BaseSRegressor.predict` prediz duas vezes —
    `hstack([zeros, X])` e depois `hstack([ones, X])`, trocando só a coluna
    de tratamento e mantendo o resto de `X` **idêntico** nas duas chamadas.
    Se colunas T×X_j fossem pré-computadas e concatenadas a `X` antes de
    passar para `BaseSRegressor`, elas ficariam fixas nas duas predições
    contrafactuais e se cancelariam exatamente na subtração — o CATE
    continuaria sendo uma constante (só o coeficiente da coluna de
    tratamento), reproduzindo silenciosamente o mesmo resultado nulo do
    S-learner linear original, sem de fato testar interação nenhuma. Por
    isso este S-learner é implementado diretamente aqui, fora do wrapper.

    Ajusta um único `ElasticNet` no design matrix `[X, T, T*X_1, ..., T*X_p]`
    — hiperparâmetros default (`alpha=1.0, l1_ratio=0.5`, os mesmos do
    S+ElasticNet original de S4.6, sem grid search) para isolar o efeito de
    adicionar interações como a única variável alterada. Preprocessing
    (features não padronizadas) também preservado idêntico ao original — ver
    ressalva sobre padronização na célula do notebook.
    """
    treatment_col = np.asarray(treatment, dtype=float).reshape(-1, 1)
    X_design = np.hstack([X, treatment_col, X * treatment_col])
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed)
    model.fit(X_design, y)
    return model


def predict_s_learner_linear_interaction_uplift(model, X):
    """CATE(x) = coef_T + Σ(coef_TXj · X_j), calculado analiticamente a partir
    dos coeficientes ajustados por `fit_s_learner_linear_interaction` — o
    modelo é linear, então μ(x,1) − μ(x,0) tem forma fechada; não é preciso
    prever duas vezes.
    """
    n_features = X.shape[1]
    coef = model.coef_
    coef_treatment = coef[n_features]
    coef_interaction = coef[n_features + 1:]
    return coef_treatment + X @ coef_interaction


def fit_causal_forest(X, treatment, y, seed=SEED):
    """Ajusta um `econml.dml.CausalForestDML` (econml 0.16.0) sobre a mesma
    matriz numérica dos meta-learners (`encode_meta_learner_features`).

    Dois parâmetros não-default, ambos exigidos para representar corretamente
    o desenho deste estudo — verificado via inspeção da API real instalada,
    não suposto de memória/documentação de outra versão (S5, item 2):

    `discrete_treatment=True` — o tratamento pooled é binário {0,1}, não
    contínuo; sem isso `CausalForestDML` regrediria T linearmente.

    `model_t=DummyClassifier(strategy='prior')` — este é um RCT: o mecanismo
    de atribuição de tratamento não depende de X (P(T=1|X) é constante por
    desenho, ver S1/S2). O default `model_t='auto'` expande, via
    `econml.dml.dml._make_first_stage_selector`/`get_selector` (inspecionado
    no código-fonte instalado), para um `GridSearchCV` sobre
    `RandomForestClassifier`/`LogisticRegressionCV` — um nuisance model
    flexível que aprenderia variação espúria de P(T=1) em função de X sob um
    mecanismo que é, por desenho, X-invariante. `DummyClassifier(strategy=
    'prior')` sempre prediz a proporção empírica de tratamento observada no
    fit, constante em X. Confirmado no código-fonte que um estimador
    sklearn-compatível passado diretamente (não `'auto'`/lista/string) cai no
    ramo `FixedModelSelector` — usado como está, sem busca por cima.

    `model_y` fica no default `'auto'` — não há incompatibilidade técnica que
    exija mudá-lo, e isso é consistente com o restante de S4, que sempre
    regride a probabilidade do outcome binário (nunca classifica).

    `random_state=seed` garante reprodutibilidade — verificado empiricamente
    (smoke test): mesmo seed produz `effect(X)` bit-a-bit idêntico.

    Custo medido em smoke test isolado, em dados no tamanho real de um
    `fit_df` de repeated holdout (~28.800 linhas × 18 colunas one-hot): ~207s
    por fit com os hiperparâmetros default (`n_estimators=100`, `cv=2`,
    `honest=True`, `n_jobs=-1`) — bem mais caro que os demais candidatos de
    S4/S5, mas computacionalmente viável; reportado como está, sem reduzir
    `n_estimators`/`cv` para acelerar (Regra Absoluta #6 — não ajustar
    hiperparâmetros que afetam o resultado só por custo). Esse número não é o
    custo representativo definitivo: na execução completa real do repeated
    holdout (15 reps × 6 candidatos, notebook 04, S5.3), o tempo total
    observado foi de ~781,5s (~13 min) — bem abaixo do que uma extrapolação
    ingênua a partir do smoke test isolado sugeriria. A causa exata da
    diferença (características dos dados reais vs. sintéticos do smoke test,
    paralelismo, condições do ambiente) não foi investigada.
    """
    model = CausalForestDML(
        model_t=DummyClassifier(strategy="prior"),
        discrete_treatment=True,
        random_state=seed,
    )
    model.fit(Y=np.asarray(y, dtype=float), T=np.asarray(treatment), X=X)
    return model


def predict_causal_forest_uplift(model, X):
    """CATE estimado: `model.effect(X)` com `T0=0`/`T1=1` (defaults da
    assinatura) — já corresponde exatamente à codificação pooled do projeto
    (controle=0, tratado=1). Retorna vetor 1D, uma estimativa por linha."""
    return np.asarray(model.effect(X)).ravel()


_UPLIFT_CONTROL_LABEL = "control"
_UPLIFT_TREATED_LABEL = "treated"


def _uplift_treatment_labels(treatment):
    """Converte o tratamento pooled 0/1 para os rótulos string exigidos pela
    API do `causalml` (`UpliftTreeClassifier`/`UpliftRandomForestClassifier.
    fit` esperam `treatment` como nomes de grupo, não 0/1 — verificado via
    `inspect.signature` e docstring do código-fonte instalado: `control_name`
    identifica explicitamente, por nome, qual rótulo é o grupo controle)."""
    treatment = np.asarray(treatment)
    return np.where(treatment == 1, _UPLIFT_TREATED_LABEL, _UPLIFT_CONTROL_LABEL)


def fit_uplift_tree(X, treatment, y, seed=SEED):
    """Ajusta um `causalml.inference.tree.UpliftTreeClassifier` (causalml
    0.15.5). Hiperparâmetros no default da biblioteca (`max_depth=3`,
    `min_samples_leaf=100`, `evaluationFunction='KL'`, sem grid search sobre
    critério/profundidade) — só `control_name` (exigência técnica da API,
    identifica o grupo controle por nome) e `random_state` (reprodutibilidade)
    são passados explicitamente."""
    labels = _uplift_treatment_labels(treatment)
    model = UpliftTreeClassifier(control_name=_UPLIFT_CONTROL_LABEL, random_state=seed)
    model.fit(X, labels, y)
    return model


def predict_uplift_tree_uplift(model, X):
    """`UpliftTreeClassifier.predict(X)` retorna, por linha, P(Y=1) para
    **cada grupo** (uma coluna por `classes_`, controle incluso) — não é
    uplift diretamente (verificado via inspeção do `.pyx` fonte instalado:
    `uplift_classification_results`/`predict`, `shape=[n_samples,
    n_treatments]` = probabilidades por grupo, não deltas). O uplift é a
    coluna do grupo tratado menos a coluna do grupo controle, localizadas via
    `classes_.index(...)` — não por posição fixa — para não presumir a ordem
    das colunas sem confirmar no source."""
    pred = model.predict(X)
    idx_treated = model.classes_.index(_UPLIFT_TREATED_LABEL)
    idx_control = model.classes_.index(_UPLIFT_CONTROL_LABEL)
    return pred[:, idx_treated] - pred[:, idx_control]


def fit_uplift_random_forest(X, treatment, y, seed=SEED):
    """Ajusta um `causalml.inference.tree.UpliftRandomForestClassifier`
    (causalml 0.15.5). Hiperparâmetros no default da biblioteca
    (`n_estimators=10`, `max_depth=5`, `min_samples_leaf=100`,
    `evaluationFunction='KL'`, `n_jobs=-1`) — só `control_name` e
    `random_state` são passados explicitamente, pelos mesmos motivos de
    `fit_uplift_tree`. Reprodutibilidade sob o `n_jobs=-1` default
    (paralelismo via threads) verificada empiricamente (smoke test): mesmo
    seed produz `predict(X)` bit-a-bit idêntico entre execuções."""
    labels = _uplift_treatment_labels(treatment)
    model = UpliftRandomForestClassifier(control_name=_UPLIFT_CONTROL_LABEL, random_state=seed)
    model.fit(X, labels, y)
    return model


def predict_uplift_random_forest_uplift(model, X):
    """`UpliftRandomForestClassifier.predict(X, full_output=True)` já calcula
    `delta_{grupo} = P(Y=1|grupo) − P(Y=1|controle)` explicitamente
    (verificado via inspeção do `.pyx` fonte instalado) — usamos
    `full_output=True` e selecionamos a coluna pelo nome exato
    (`delta_treated`), em vez de confiar na ordem posicional do array
    retornado por `full_output=False` (que também é o uplift, mas sem rótulo
    de coluna — não escolhido "porque parece certo", confirmado no source)."""
    df_res = model.predict(X, full_output=True)
    return df_res[f"delta_{_UPLIFT_TREATED_LABEL}"].to_numpy()
