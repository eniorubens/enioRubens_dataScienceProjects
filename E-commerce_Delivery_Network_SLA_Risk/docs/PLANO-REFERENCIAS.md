# Plano de citações e referências nos notebooks

**Estado:** **aprovado em 2026-11-16** (P-08 em `REGISTRO-DE-DECISOES.md`), incluindo o item opcional da seção 3. Aplicado em 2026-11-16 aos notebooks 01, 03 a 06 e 08 a 10. O notebook 07 recebeu as citações depois do fim da sua execução, no mesmo dia. Bergstra et al. (2011) não foi citado, porque nenhum texto do notebook 08 menciona o TPE. Sinnott (1984) foi confirmado só por fontes secundárias, e o ano de Olist (2018) não aparece na página do Kaggle.
**Origem:** pedido do responsável para dar ao leitor a origem das definições e fórmulas usadas. A pergunta 24 do anexo (padrão bibliográfico) já foi aprovada como autor-data simples (P-07).

## 1. Regras

1. **Formato da citação no texto:**
   - um autor: (Sinnott, 1984);
   - dois autores: (Breusch e Pagan, 1979) em PT e (Breusch and Pagan, 1979) em EN;
   - três ou mais autores: (Akiba et al., 2019).
2. **Onde citar:** a citação entra na primeira menção do método em cada notebook, dentro de uma frase que já existe. Nenhuma afirmação nova é criada para justificar uma citação. A única exceção é o item opcional da seção 3.
3. **Lista de referências:** cada notebook ganha uma última célula Markdown, "## Referências" (PT) ou "## References" (EN). Ela traz, em ordem alfabética, apenas as obras citadas naquele notebook, no formato `Sobrenome, I. (ano). Título. Veículo, volume(número), páginas.` Livros trazem a editora.
4. **Paridade:** o PT e o EN citam as mesmas obras nos mesmos lugares. Só mudam o conectivo "e"/"and" e a narrativa.
5. **Execução:** só células Markdown mudam, então nenhum notebook precisa ser reexecutado. A narrativa segue em voz passiva e sem bullets.
6. **Notebook 07:** só é editado depois que a execução em curso terminar.
7. **Conferência obrigatória:** cada referência é conferida (autores, ano, título, veículo, volume e páginas) antes de entrar. Uma referência cujo detalhe não puder ser confirmado entra sem o detalhe incerto (por exemplo, sem páginas) e nunca com um valor suposto.
8. **Teste de contrato novo:**
   - toda citação "(Autor, ano)" do texto tem entrada na lista de referências do notebook, e toda entrada é citada;
   - o PT e o EN têm o mesmo conjunto de referências.

## 2. Mapa de citações por notebook

| Notebook | Seção (PT) | Conceito no texto | Citação proposta |
|---|---|---|---|
| 01 | Abertura e inventário | Fonte de dados | (Olist, 2018) |
| 02 | — | Sem definição ou fórmula metodológica; a fonte já é citada no 01 | Nenhuma |
| 03 | "O que a geolocalização permite medir?" | Fórmula de Haversine | (Sinnott, 1984) |
| 03 | mesma seção | Raio médio da Terra de 6.371,0088 km (raio médio do elipsoide GRS80) | (Moritz, 2000) |
| 04 | "Uma linha por pedido" | Distância Haversine | (Sinnott, 1984) |
| 05 | "Folds expanding-window de desenvolvimento" | Validação temporal com origem móvel e janela expansiva | (Tashman, 2000; Bergmeir e Benítez, 2012) |
| 05 | mesma seção | Embargo entre treino e validação | (López de Prado, 2018) |
| 05 | "Deriva das features entre treino e validação" | PSI e limiares convencionais 0,10 e 0,25 | (Siddiqi, 2006; Yurdakul, 2018) |
| 05 | "Sensibilidade à censura à direita" | Censura à direita | (Klein e Moeschberger, 2003) |
| 06 | "Preparação sem vazamento" | Ridge | (Hoerl e Kennard, 1970) |
| 06 | mesma seção | Regressão logística | (Cox, 1958) |
| 06 | "Desempenho preditivo..." | Average precision | (Davis e Goadrich, 2006; Saito e Rehmsmeier, 2015) |
| 06 | mesma seção | Brier | (Brier, 1950) |
| 06 | mesma seção | ECE | (Naeini et al., 2015) |
| 06 | "Diagnóstico inferencial do OLS" | Erros-padrão HC3 | (White, 1980; MacKinnon e White, 1985) |
| 06 | mesma seção | Erros-padrão por cluster | (Cameron e Miller, 2015) |
| 06 | mesma seção | VIF | (Marquardt, 1970) |
| 06 | mesma seção | Teste de Breusch–Pagan | (Breusch e Pagan, 1979) |
| 06 | mesma seção | Distância de Cook | (Cook, 1977) |
| 07 | "Mesmo protocolo" | Validação temporal expansiva | (Bergmeir e Benítez, 2012) |
| 07 | mesma seção | Calibração isotônica | (Zadrozny e Elkan, 2002; Niculescu-Mizil e Caruana, 2005) |
| 07 | mesma seção | scikit-learn (`CalibratedClassifierCV`, `StandardScaler`) | (Pedregosa et al., 2011) |
| 07 | Insight e "Critério de decisão" | AP, ROC AUC, Brier e ECE | (Davis e Goadrich, 2006; Fawcett, 2006; Brier, 1950; Naeini et al., 2015) |
| 07 | Primeira menção de cada estimador no Markdown, se houver | CatBoost, XGBoost, LightGBM, random forest, extra trees, gradient boosting | (Prokhorenkova et al., 2018), (Chen e Guestrin, 2016), (Ke et al., 2017), (Breiman, 2001), (Geurts et al., 2006), (Friedman, 2001) |
| 08 | Abertura e "Busca reprodutível..." | Optuna; amostrador TPE (`TPESampler`) | (Akiba et al., 2019; Bergstra et al., 2011) |
| 08 | mesma seção | ElasticNet, Ridge e calibração isotônica | (Zou e Hastie, 2005), (Hoerl e Kennard, 1970), (Zadrozny e Elkan, 2002) |
| 09 | "Pré-registro" | Pré-registro | (Nosek et al., 2018) |
| 09 | mesma tabela | CatBoost | (Prokhorenkova et al., 2018) |
| 09 | mesma tabela | AP, ROC AUC, Brier e ECE | (Davis e Goadrich, 2006; Fawcett, 2006; Brier, 1950; Naeini et al., 2015) |
| 09 | mesma tabela | Curva de confiabilidade | (DeGroot e Fienberg, 1983) |
| 09 | mesma tabela | Bootstrap por blocos e IC percentil | (Künsch, 1989; Efron e Tibshirani, 1993) |
| 09 | "Drift entre treino e teste" | PSI | (Siddiqi, 2006) |
| 10 | Abertura e seção 1 | Reuso do período de teste após observá-lo | (Dwork et al., 2015) |
| 10 | Seção 2 | Rótulos que chegam com atraso | (Chapelle, 2014) |
| 10 | Seção 2 | Bootstrap por blocos | (Künsch, 1989) |
| 10 | Seção 3 | Pesos por recência e mudança de regime | (Gama et al., 2014) |
| 10 | Seção 3 | Recalibração isotônica | (Zadrozny e Elkan, 2002) |
| 10 | Seção 5 | Ajuste para múltiplas comparações (não aplicado) | (Benjamini e Hochberg, 1995) |
| 10 | Seção 6 | Tratar pedidos em trânsito como censurados | (Kaplan e Meier, 1958) |

## 3. Item opcional: definição das taxas históricas da rede

Hoje nenhum notebook define em texto como as taxas históricas da rede são suavizadas. A definição existe só em `network.py`: a taxa observada é puxada para a taxa global as-of com força fixa m = 20, pela fórmula (soma dos eventos + m × prior) / (n + m). A proposta é acrescentar essa definição em uma frase no notebook 04, na seção do catálogo, citando (Cestnik, 1990; Micci-Barreca, 2001). É conteúdo novo, por isso fica como opção separada.

## 4. Referências propostas

As entradas abaixo são a base a conferir. Detalhes não confirmados serão omitidos, conforme a regra 7.

| Citação | Referência |
|---|---|
| Akiba et al., 2019 | Akiba, T., Sano, S., Yanase, T., Ohta, T. e Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD*, 2623–2631. |
| Benjamini e Hochberg, 1995 | Benjamini, Y. e Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289–300. |
| Bergmeir e Benítez, 2012 | Bergmeir, C. e Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192–213. |
| Bergstra et al., 2011 | Bergstra, J., Bardenet, R., Bengio, Y. e Kégl, B. (2011). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems 24*. |
| Breiman, 2001 | Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. |
| Breusch e Pagan, 1979 | Breusch, T. S. e Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287–1294. |
| Brier, 1950 | Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. |
| Cameron e Miller, 2015 | Cameron, A. C. e Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372. |
| Cestnik, 1990 | Cestnik, B. (1990). Estimating probabilities: a crucial task in machine learning. *Proceedings of ECAI 1990*, 147–149. |
| Chapelle, 2014 | Chapelle, O. (2014). Modeling delayed feedback in display advertising. *Proceedings of the 20th ACM SIGKDD*, 1097–1105. |
| Chen e Guestrin, 2016 | Chen, T. e Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794. |
| Cook, 1977 | Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15–18. |
| Cox, 1958 | Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society B*, 20(2), 215–242. |
| Davis e Goadrich, 2006 | Davis, J. e Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the 23rd ICML*, 233–240. |
| DeGroot e Fienberg, 1983 | DeGroot, M. H. e Fienberg, S. E. (1983). The comparison and evaluation of forecasters. *The Statistician*, 32(1/2), 12–22. |
| Dwork et al., 2015 | Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. e Roth, A. (2015). The reusable holdout: preserving validity in adaptive data analysis. *Science*, 349(6248), 636–638. |
| Efron e Tibshirani, 1993 | Efron, B. e Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. |
| Fawcett, 2006 | Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. |
| Friedman, 2001 | Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. |
| Gama et al., 2014 | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. e Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 44. |
| Geurts et al., 2006 | Geurts, P., Ernst, D. e Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42. |
| Hoerl e Kennard, 1970 | Hoerl, A. E. e Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. |
| Kaplan e Meier, 1958 | Kaplan, E. L. e Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457–481. |
| Ke et al., 2017 | Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. e Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*. |
| Klein e Moeschberger, 2003 | Klein, J. P. e Moeschberger, M. L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data* (2ª ed.). Springer. |
| Künsch, 1989 | Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217–1241. |
| López de Prado, 2018 | López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. |
| MacKinnon e White, 1985 | MacKinnon, J. G. e White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325. |
| Marquardt, 1970 | Marquardt, D. W. (1970). Generalized inverses, ridge regression, biased linear estimation, and nonlinear estimation. *Technometrics*, 12(3), 591–612. |
| Micci-Barreca, 2001 | Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations*, 3(1), 27–32. |
| Moritz, 2000 | Moritz, H. (2000). Geodetic Reference System 1980. *Journal of Geodesy*, 74(1), 128–133. |
| Newman, 2010 | Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press. |
| Naeini et al., 2015 | Naeini, M. P., Cooper, G. F. e Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the 29th AAAI Conference*, 2901–2907. |
| Niculescu-Mizil e Caruana, 2005 | Niculescu-Mizil, A. e Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd ICML*, 625–632. |
| Nosek et al., 2018 | Nosek, B. A., Ebersole, C. R., DeHaven, A. C. e Mellor, D. T. (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606. |
| Olist, 2018 | Olist (2018). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle. |
| Pedregosa et al., 2011 | Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. |
| Prokhorenkova et al., 2018 | Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. e Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems 31*. |
| Saito e Rehmsmeier, 2015 | Saito, T. e Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432. |
| Siddiqi, 2006 | Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley. |
| Sinnott, 1984 | Sinnott, R. W. (1984). Virtues of the Haversine. *Sky and Telescope*, 68(2), 159. |
| Tashman, 2000 | Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437–450. |
| White, 1980 | White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838. |
| Yurdakul, 2018 | Yurdakul, B. (2018). *Statistical Properties of Population Stability Index*. Tese de doutorado, Western Michigan University. |
| Zadrozny e Elkan, 2002 | Zadrozny, B. e Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of the 8th ACM SIGKDD*, 694–699. |
| Zou e Hastie, 2005 | Zou, H. e Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society B*, 67(2), 301–320. |

## 5. Execução concluída

1. As referências foram conferidas conforme a regra 7; detalhes sem confirmação foram omitidos.
2. As citações e células de referências foram inseridas nos notebooks 01, 03 a 06 e 08 a 10; o notebook 07 foi atualizado depois da execução do torneio.
3. O teste de contrato da regra 8 foi acrescentado.
4. Os testes de contrato e o lint foram executados com sucesso.
5. A edição foi feita com os notebooks fechados no VS Code para evitar sobrescrita de conteúdo.
