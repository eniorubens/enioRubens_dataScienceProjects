# Dicionário de Termos Técnicos — Projeto Uplift Modeling

> Glossário de referência para o projeto Hillstrom Uplift Modeling. Organizado por área temática. Termos marcados com ⭐ apareceram explicitamente no notebook até a Seção 2.

---

## 1. Marketing e E-mail Marketing

| Termo | Definição |
|---|---|
| ⭐ **Fadiga de base (list fatigue)** | Fenômeno em que clientes recebem comunicações em excesso, passam a ignorar, marcar como spam ou cancelar inscrição. Cada envio "queima" um pouco da disposição da base — é por isso que e-mail não é "de graça". |
| ⭐ **Opt-out** | Ato do cliente sair da lista de comunicações (clicar em "descadastrar"). Métrica vigiada de perto, porque um opt-out é perda permanente do canal direto com aquele cliente. |
| **Opt-in** | O inverso: o cliente autoriza explicitamente receber comunicações. Lei brasileira (LGPD) exige opt-in para muitas categorias. |
| ⭐ **Deliverability** | Capacidade de o e-mail efetivamente chegar à caixa de entrada (não cair em spam, lixeira, ou ser bloqueado pelo provedor). Provedores tipo Gmail e Outlook penalizam senders com baixo engajamento. |
| **Sender reputation** | "Score" que provedores de e-mail (Gmail, Outlook) atribuem ao remetente. Baixa reputação = mais e-mails caem em spam = deliverability cai = ROI cai. Espiral negativa. |
| **Bounce rate** | Taxa de e-mails que voltam. *Hard bounce* = endereço inexistente (permanente); *soft bounce* = caixa cheia, problema temporário. |
| ⭐ **Top of funnel (TOFU) / Bottom of funnel (BOFU)** | Estágios do funil de compra. TOFU = primeiro contato, awareness, visita ao site (alta volumetria, baixa conversão). BOFU = decisão de compra (baixa volumetria, alta conversão). No nosso projeto, `visit` é TOFU; `conversion` é BOFU. |
| ⭐ **Funnel de conversão** | Sequência de estágios entre o primeiro contato e a compra. Cada passo perde gente — daí "funil" (largo no topo, estreito embaixo). |
| ⭐ **ROI marginal** | Retorno sobre investimento **incremental** de uma decisão específica. Diferente do ROI médio. Tratá-lo como sinônimo de ROI médio é o erro número um em análise de marketing. |
| **ROAS** | Return on Ad Spend. Receita gerada dividida pelo custo da campanha. Métrica mais usada no operacional do que ROI. |
| ⭐ **Targeting** | Decisão de a quem dirigir uma campanha. O *core business* deste projeto é melhorar targeting via uplift modeling em vez de propensity. |
| **Retargeting** | Mostrar anúncio/campanha para quem já interagiu com a marca antes (visitou site, abriu e-mail). Diferente de prospecting (alcançar quem ainda não conhece). |
| **CTR (Click-Through Rate)** | Cliques / impressões. Mede engajamento com o e-mail/anúncio. Não confundir com conversão. |
| ⭐ **Conversion rate** | Conversões / impressões (ou conversões / visitas, depende do denominador). No Hillstrom, é compras dividido por clientes contatados. |
| ⭐ **Lift** | Diferença de performance entre um grupo tratado e um grupo de controle. Usado solto, é sinônimo informal de ATE. Não confundir com uplift, que é lift **por indivíduo** (CATE). |
| ⭐ **Uplift** | Efeito incremental do tratamento estimado em nível individual. *A diferença entre lift e uplift é a diferença entre "média da campanha" e "efeito em cada cliente".* |
| **Engagement** | Métrica composta de interação: abriu o e-mail, clicou, respondeu, visitou. Várias indústrias definem de forma diferente. |
| **CLTV / LTV (Customer Lifetime Value)** | Receita esperada de um cliente ao longo de toda a relação comercial. Métrica-base para decisões de aquisição e retenção. |
| **Churn** | Cancelamento de cliente (assinaturas) ou inatividade prolongada (varejo). Você já fez um projeto inteiro sobre isso. |
| **NBA (Next Best Action)** | Recomendação automatizada da próxima ação ótima a tomar com um cliente (qual produto oferecer, qual canal usar, qual oferta). Você já fez um projeto sobre isso também. |

---

## 2. Desenho Experimental e Estatística

| Termo | Definição |
|---|---|
| ⭐ **RCT (Randomized Controlled Trial)** | Experimento onde a atribuição de tratamento é aleatória. Padrão-ouro para inferência causal porque a randomização elimina confounders. O Hillstrom é um RCT. |
| **A/B test** | RCT na linguagem de tech/produto. Mesma coisa, vocabulário diferente. |
| ⭐ **Braço (arm)** | Cada grupo de tratamento no experimento. Hillstrom tem 3 braços: No E-Mail, Mens E-Mail, Womens E-Mail. |
| ⭐ **Grupo de controle (control)** | Braço que NÃO recebe o tratamento. Serve como contrafactual para estimar o efeito. |
| ⭐ **Randomização** | Atribuição aleatória de unidades a braços. É o que garante que os grupos sejam comparáveis em média. *Sem randomização (ou sem assumir randomização), não há inferência causal limpa.* |
| ⭐ **Balance / balanceamento** | Propriedade de que as covariáveis estejam distribuídas de forma similar entre os braços. Consequência esperada da randomização — quando não acontece, sinaliza problema no design. |
| **Estratificação** | Randomização *dentro* de blocos (e.g., randomizar separadamente entre clientes novos e antigos) para garantir balance perfeito em variáveis-chave. |
| ⭐ **Covariável pré-tratamento** | Variável medida **antes** do tratamento ser aplicado. Crucial: variáveis medidas pós-tratamento podem ser efeito do tratamento e introduzem viés se usadas como features. |
| ⭐ **Intervalo de confiança (IC)** | Faixa de valores plausíveis para um parâmetro estimado, dado um nível de confiança (geralmente 95%). Interpretação técnica: se repetirmos o experimento muitas vezes, em 95% das vezes o IC vai conter o valor real. Interpretação prática: "o efeito real está provavelmente entre X e Y". |
| ⭐ **p-value** | Probabilidade de observar um resultado tão ou mais extremo que o observado, *assumindo que H₀ é verdadeira*. Convenção: p < 0.05 → rejeitar H₀. Não é a probabilidade de H₀ ser verdadeira (erro comum). |
| **Hipótese nula (H₀)** | A hipótese de "nada acontece" / "não há diferença". Em RCT, H₀ típica é ATE = 0. |
| **Significância estatística** | Rejeição formal de H₀. Não é o mesmo que significância **prática** — com N grande, qualquer diferença minúscula vira "estatisticamente significativa" sem ter valor de negócio. |
| **Tamanho de efeito (effect size)** | Magnitude da diferença observada, em unidades interpretáveis. Diferente de significância: é "quanto" em vez de "tem ou não tem". SMD é um effect size. |
| ⭐ **SMD (Standardized Mean Difference)** | Diferença de médias dividida pelo desvio-padrão pooled. Métrica padrão de balance. \|SMD\| < 0.1 = balance aceitável; < 0.05 = excelente. É insensível a tamanho de amostra (vantagem sobre p-value). |
| ⭐ **ANOVA / F-test** | Teste estatístico para comparar médias entre 3+ grupos. Pergunta: "as médias são todas iguais?". Usei na S2 para balance check das contínuas. |
| ⭐ **Qui-quadrado (χ², chi-squared)** | Teste para verificar associação entre duas variáveis categóricas. Pergunta: "a distribuição é independente do grupo?". Usei na S2 para balance check das categóricas. |
| ⭐ **Welch t-test** | Variante do t-test que não assume variâncias iguais entre os grupos. Mais robusto que o t-test clássico (Student). |
| **Erro-padrão (SE)** | Desvio-padrão da distribuição amostral de um estimador. "Quão precisa é a estimativa?". IC = estimativa ± z × SE. |
| ⭐ **Classe rara (class imbalance)** | Quando um outcome binário tem prevalência muito baixa (< 5%, digamos). Caso de `conversion` no Hillstrom (0.9%). Complica modelagem porque o modelo pode aprender a prever sempre "0" e ter alta acurácia inútil. |
| **Powering / power analysis** | Cálculo do tamanho de amostra necessário para detectar um efeito de tamanho X com probabilidade Y. Importante antes do experimento; menos relevante depois (que é o nosso caso). |
| **Attrition** | Perda de unidades do experimento ao longo do tempo (clientes que somem, opt-outs, dados perdidos). Attrition **diferencial** entre braços viola randomização efetiva. |
| ⭐ **Forest plot** | Visualização de múltiplas estimativas pontuais com seus ICs, em formato de bandeirinhas horizontais. Padrão em meta-análises e RCTs. Usei na S2.5 para os ATEs. |
| ⭐ **Love plot** | Plot canônico de balance check em RCTs/observacionais. Mostra SMDs de todas as covariáveis com linhas verticais nos thresholds aceitáveis. Nomeado em homenagem a Thomas Love, não pelo sentimento. |
| **Holdout / test set** | Subconjunto de dados separado, **não usado** no treino, para avaliação honesta do modelo. Sem holdout, qualquer métrica vira mentira (overfitting). |
| **Validação cruzada (cross-validation)** | Técnica de avaliar modelo treinando em K fatias diferentes dos dados. Padrão: K=5 ou K=10. |

---

## 3. Causal Inference

| Termo | Definição |
|---|---|
| ⭐ **Potential outcomes (resultados potenciais)** | Framework formal: para cada indivíduo, existem dois "mundos paralelos" — $Y_i(1)$ se tratado, $Y_i(0)$ se não tratado. Só observamos um deles. Base de toda a teoria moderna de inferência causal (Rubin, 1974). |
| **Counterfactual** | O resultado em um cenário que **não aconteceu**. "O que teria acontecido se...". É o resultado potencial não observado. |
| ⭐ **ITE (Individual Treatment Effect)** | $\tau_i = Y_i(1) - Y_i(0)$. O efeito do tratamento para um indivíduo específico. Inobservável por princípio. |
| ⭐ **ATE (Average Treatment Effect)** | $E[Y(1) - Y(0)]$. Efeito médio na população. Estimável em RCT pela diferença de médias entre braços. |
| ⭐ **CATE (Conditional Average Treatment Effect)** | $\tau(x) = E[Y(1) - Y(0) \mid X=x]$. Efeito médio para um *subgrupo* definido por covariáveis $X$. **É o que uplift modeling estima.** |
| ⭐ **Heterogeneidade de tratamento** | Variação do efeito entre indivíduos/subgrupos. Quando $\tau(x)$ varia muito com $x$, faz sentido fazer targeting personalizado. Quando é constante, não. |
| ⭐ **Unconfoundedness / ignorability** | Assumption central: $\{Y(0), Y(1)\} \perp T \mid X$. Significa "depois de controlar pelas covariáveis observadas, o tratamento é tão bom quanto aleatório". Garantida automaticamente em RCT; precisa ser defendida em observacional. |
| **SUTVA (Stable Unit Treatment Value Assumption)** | Assumption: o tratamento de um indivíduo não afeta o resultado de outro (no spillover), e há uma única "versão" do tratamento. Quebrada por exemplo se você manda e-mail e o cliente conta para a família que também compra. |
| **Confounder / confounding** | Variável que afeta tanto o tratamento quanto o outcome, criando associação espúria. Clássico: clientes mais ricos recebem mais e-mails E compram mais — sem controlar por renda, o efeito do e-mail fica inflado. |
| ⭐ **Identificação causal** | Condição em que o efeito causal é estimável a partir dos dados observados. Não é questão estatística (mais dados não resolve), é questão estrutural do design. |
| **Propensity score** | $P(T=1 \mid X)$. Probabilidade de ser tratado dadas as covariáveis. Em RCT, propensity é constante (~1/3 em 1:1:1); em observacional, varia. Usado para ajustar viés. |
| **IPW (Inverse Probability Weighting)** | Técnica de ponderar observações pelo inverso da propensity, criando "pseudo-RCT" a partir de dados observacionais. Sensível a propensities extremas. |
| **Doubly robust** | Família de estimadores que combinam modelo de outcome + modelo de propensity. Vantagem: se **um dos dois** estiver correto, a estimativa é consistente. DR-learner é exemplo. |
| **Quasi-experimental** | Design que não é RCT mas aproveita variação "como se" aleatória (regression discontinuity, diff-in-diff, instrumental variables). |
| **Observational study** | Estudo sem manipulação ativa de tratamento. Mais barato e mais comum, mas exige assumptions mais fortes para inferência causal. |
| **Quadrantes de resposta** | Persuadables / Sure Things / Lost Causes / Sleeping Dogs. Detalhados no glossário do notebook na S1.3. |

---

## 4. Modelos de Uplift e Avaliação

| Termo | Definição |
|---|---|
| ⭐ **Meta-learner** | Estratégia de estimar CATE usando algoritmos de ML padrão (regressão, classificação) como blocos. Não inventa algoritmo novo, *orquestra* algoritmos existentes. Variantes: S, T, X, R, DR. |
| **S-learner** | "Single" — um único modelo prevê $Y$ usando $X$ e $T$ como features. CATE = $\hat{Y}(X, T=1) - \hat{Y}(X, T=0)$. Simples mas pode subestimar uplift se o modelo ignorar $T$. |
| **T-learner** | "Two" — dois modelos separados, um por braço. CATE = diferença das predições. Mais flexível que S-learner mas perde eficiência (não compartilha dados entre braços). |
| **X-learner** | Variante mais sofisticada que combina T-learner com propensity weighting. Boa quando os braços têm tamanhos muito desiguais. |
| **R-learner** | Usa resíduos da regressão de Y em X e de T em X. Tem boas propriedades teóricas (Nie & Wager, 2021). |
| **Direct uplift model** | Algoritmo treinado especificamente para estimar uplift, não adaptado de classificação/regressão. Causal Forest e Uplift Trees são exemplos. |
| ⭐ **Causal Forest** | Variante de Random Forest projetada para estimar CATE com intervalos de confiança. Splits da árvore otimizam diferença de tratamento, não acurácia de outcome. (Athey & Wager, 2019.) |
| **Uplift Random Forest** | Outra família de árvore para uplift, com critérios de split específicos (KL-divergence, Euclidean distance, chi-squared). Implementação principal: CausalML (Uber). |
| **Honest splitting** | Técnica em árvores causais: separar a amostra usada para escolher splits da amostra usada para estimar valores nas folhas. Reduz overfitting e habilita intervalos de confiança válidos. |
| ⭐ **Qini curve** | Visualização de avaliação de uplift. Eixo x: % da população tratada, ordenada do maior uplift score para o menor. Eixo y: número incremental de positivos capturados. Quanto mais curvada para cima, melhor o modelo. |
| ⭐ **Qini coefficient** | Área entre a Qini curve e a linha de baseline aleatório. Análogo do Gini para uplift. Métrica número 1 da literatura. |
| ⭐ **Uplift curve / AUUC** | Variante da Qini que usa percentuais em vez de contagens. AUUC = Area Under the Uplift Curve. |
| ⭐ **Uplift@k** | Uplift estimado para os top-k% segundo o modelo. Métrica direta de política: "se eu só puder tratar 10% da base, qual o uplift médio nesse 10%?". |
| **Policy learning** | Disciplina de aprender uma **política de decisão** (quem tratar, com qual tratamento) diretamente dos dados, em vez de só estimar CATE e fazer threshold depois. (Athey & Wager 2021; Kitagawa & Tetenov 2018.) |
| **Targeting policy** | A política de decisão resultante. "Trate todo cliente com `recency <= 3` e `mens=1`" é uma política. |

---

## 5. Termos operacionais de ML que aparecem no projeto

| Termo | Definição |
|---|---|
| **Pipeline** | Sequência reproduzível de etapas: load → preprocess → feature engineering → train → evaluate → predict. Você tem o seu, `optpipe`. |
| ⭐ **MLflow** | Ferramenta de tracking de experimentos: registra hiperparâmetros, métricas, artefatos (modelos serializados), versão do código. Permite comparar runs lado a lado. |
| ⭐ **Optuna** | Framework de otimização de hiperparâmetros usando algoritmos como TPE (Tree-structured Parzen Estimator). Mais inteligente que grid search ou random search. |
| **Tuning / tunagem** | Processo de buscar a melhor combinação de hiperparâmetros para um modelo. |
| ⭐ **Hiperparâmetros** | Configurações do algoritmo que **não** são aprendidas dos dados — você escolhe. Ex.: profundidade máxima de uma árvore, learning rate de um booster, número de árvores numa floresta. |
| **Calibração** | Propriedade de um modelo probabilístico cujas probabilidades preditas refletem frequências reais. Modelo bem calibrado: quando prevê 70%, acerta 70% do tempo. |
| ⭐ **SHAP (SHapley Additive exPlanations)** | Framework de interpretabilidade que atribui contribuição de cada feature à predição de cada observação. Baseado em teoria de jogos cooperativos (valores de Shapley). |
| ⭐ **Boosters (XGBoost, LightGBM)** | Famílias de modelos de gradient boosting (sequência de árvores em que cada nova árvore corrige erros das anteriores). Quase sempre o estado da arte em dados tabulares. |
| **Serialização** | Salvar modelo treinado em arquivo para reuso (formato pickle, joblib, ou nativo da lib). Habilita o seu padrão `RETRAIN = False`. |

---

## Sugestões para você expandir

Pensa em adicionar termos que sejam relevantes para a **leitura do notebook por outra pessoa** — recrutador, colega, você daqui a 6 meses. Possíveis lacunas que eu deixei propositalmente fora porque são padrão demais (mas podem valer):

- Termos básicos de Python/Pandas (DataFrame, Series, groupby, etc.)
- Termos básicos de ML (overfitting, regularização, ROC, AUC)
- Termos específicos do seu stack (`ds_toolkit`, `multilang`, `optpipe`)

Me diz o que falta ou o que ficou confuso e eu refino.
