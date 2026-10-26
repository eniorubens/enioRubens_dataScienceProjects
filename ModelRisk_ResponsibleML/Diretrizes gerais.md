Sugestão de nome:

**Model Risk & Responsible ML — Auditoria reproduzível no UCI Adult**

Pasta:

`D:\Projetos\ModelRisk_ResponsibleML`

Nome mais específico, se preferir:

**ResponsibleML — Auditoria integrada de risco, calibração e fairness**

## Escopo básico para o novo projeto

**Objetivo**

Construir uma auditoria reproduzível de risco, calibração e fairness usando o conjunto UCI Adult, comparando modelos sob uma perspectiva integrada de desempenho, incerteza, explicabilidade e limitações de validade.

O projeto não pretende ser o primeiro estudo de fairness no UCI Adult. O dataset já foi amplamente utilizado em trabalhos sobre classificação justa, equalized odds, calibração, mitigação de vieses e explicabilidade. O diferencial proposto é organizar essas dimensões em um protocolo único, rastreável e orientado a risco, com separação rigorosa entre desenvolvimento, calibração e teste.

Fonte oficial: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

**Pergunta central**

> Um modelo aparentemente bom globalmente continua aceitável quando analisamos seus erros, probabilidades e decisões por grupo?

**Posicionamento da proposta**

> O melhor compromisso entre desempenho, incerteza, calibração, explicabilidade e diferenças entre grupos permanece aceitável quando consideramos também a validade externa e as limitações históricas do UCI Adult?

**Modelos candidatos**

- DummyClassifier como referência mínima;
- Logistic Regression como baseline interpretável;
- Random Forest;
- XGBoost ou HistGradientBoosting;
- SVC como candidato orientado à precisão.

**Protocolo**

- Separação estratificada entre treino, calibração e teste;
- Nenhuma decisão de limiar usando o conjunto de teste;
- Comparação global e por grupo;
- Reprodutibilidade por seed, versão do dataset e fingerprint dos dados;
- Avaliação com intervalos de confiança quando aplicável.

**Métricas globais**

- ROC-AUC e PR-AUC;
- Accuracy e balanced accuracy;
- Precision, recall e F1;
- Brier score;
- Curva de calibração;
- Matriz de confusão;
- Desempenho em diferentes limiares.
- Intervalos de confiança ou outra medida explícita de incerteza para métricas e diferenças relevantes.

**Auditoria de grupos**

Usar inicialmente sexo e raça como atributos de auditoria, mesmo que não entrem nas features preditivas:

- taxa de seleção;
- recall/TPR;
- FPR;
- FNR;
- precision/PPV;
- diferenças e razões entre grupos;
- análise de interseções, como sexo × raça.
- incerteza das estimativas por grupo e estabilidade das conclusões;
- distinção entre disparidade observada, desempenho preditivo e alegação causal;
- discussão de atributos proxy e do efeito de excluir ou manter atributos de auditoria.

**Explicabilidade**

- importância global e, quando aplicável, explicações locais;
- análise de coerência entre explicabilidade, desempenho e diferenças entre grupos;
- explicações tratadas como evidência auxiliar, não como prova de causalidade ou ausência de viés.

**Limitações de validade**

- documentar que o UCI Adult é derivado de dados censitários de 1994;
- registrar que o limiar de renda de US$ 50 mil pode representar percentis muito diferentes entre grupos;
- não generalizar os resultados para decisões atuais, pessoas reais ou outros contextos sem validação adicional;
- considerar, em etapa futura ou análise de sensibilidade, referências como Folktables/ACS para discutir estabilidade e distribuição.

**Entregáveis**

1. Notebook de contrato e qualidade dos dados.
2. Notebook de baseline e comparação dos modelos.
3. Notebook de calibração e escolha de limiar.
4. Notebook de auditoria de fairness e risco.
5. Model card do modelo escolhido.
6. Relatório executivo com limitações e recomendações.
7. Arquivos agregados para eventual dashboard no Power BI.
8. Suíte mínima de testes para impedir vazamento e inconsistências.
9. Notebooks devem ser modulares com o mínimo de código possível, os fontes 
devem estar em .\src\*.py com docstring PEP8.
10. Pasta para notebooks em Português : .\pt-BR ; e Inglês : .\en-US
11. Os scripts em Python para apresentação em pt-BR (canônico) e ingles-US,
usando os módulos multilang (D:\Projetos\ds_toolkit\multilang)
12. O storytelling dos notebooks deve ser em linguagem simples para pessoas não técnicas entenderem. Contudo, as citações 
devem preservar linguagem técnica apropriada, mas elucidada no parágrafo seguinte para todos leitores entenderem.
13. Quando os dados apresenta alguma elucidação ou achado, apresente storytelling encabeçado por 'Insights: '
14. Evite pointbullets, abra um parágrafo e discorra o achado. 

**Estrutura obrigatória das seções com `Insights:`**

Cada seção analítica dos notebooks deve seguir esta sequência narrativa:

1. **Título:** nome claro e específico do problema analisado.
2. **Apresentação do problema:** explicitar a pergunta que a seção pretende responder e por que ela importa para o projeto.
3. **Elucidação das técnicas utilizadas:** explicar, em linguagem acessível, quais estatísticas, comparações, métricas ou procedimentos serão usados e quais são suas limitações.
4. **Código Python:** apresentar o código mínimo necessário, preferencialmente chamando funções reutilizáveis de `src`.
5. **Evidência:** mostrar as tabelas, métricas, gráficos ou outros resultados produzidos pelo código.
6. **`Insights:`:** encerrar a seção com a interpretação do achado observado, retomando explicitamente a pergunta inicial em um ou mais parágrafos corridos, sem lista de tópicos. Usar um único cabeçalho `Insights:` por seção; não repetir `Insight:` no início de cada parágrafo.

O bloco `Insights:` não deve antecipar a evidência nem substituir a explicação da técnica. Deve distinguir associação descritiva, resultado preditivo, incerteza e alegação causal, conforme aplicável. Quando os dados não permitirem uma resposta conclusiva, o bloco deve declarar a limitação e indicar a próxima análise necessária.

**Fora do escopo**

- Deploy produtivo;
- decisão automatizada sobre pessoas reais;
- alegações jurídicas de conformidade;
- retreinamento automático;
- conclusão de que remover atributos protegidos elimina viés;
- maximização de uma métrica única.

**Critério de sucesso**

O projeto será considerado concluído quando conseguirmos justificar, com evidências e incerteza explicitada, qual modelo oferece o melhor compromisso entre desempenho, calibração, interpretabilidade e diferenças entre grupos, considerando as limitações de validade do UCI Adult — ou concluir que nenhum candidato é aceitável sem mitigação adicional.

Para iniciar o novo chat, você pode colar:

> Desenvolver o projeto **Model Risk & Responsible ML** em `D:\Projetos\ModelRisk_ResponsibleML`, usando o UCI Adult. O foco é comparar modelos e auditar risco, calibração e fairness por sexo e raça. Seguir protocolo reproduzível, sem vazamento, com notebooks como narrativa, lógica reutilizável em `src`, resultados agregados e documentação das limitações. Começar pelo contrato dos dados e pelo baseline interpretável.
