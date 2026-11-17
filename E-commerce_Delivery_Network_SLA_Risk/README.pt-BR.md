# Rede de entrega e risco de SLA no e-commerce (Olist)

Este projeto investiga o atraso de entrega e o risco de violação do prazo prometido (SLA) em pedidos do marketplace brasileiro Olist. O código reutilizável fica no pacote ecom_delivery_sla, e os gráficos de rede e apresentação ficam no pacote ecom_delivery_sla_viz.

O estado do projeto é **exploratório**: production_eligible = false. Nenhum modelo foi considerado elegível para uso operacional.

## Objetivo

A pergunta central é se, no momento da compra, o risco de um pedido ser entregue depois da data prometida pode ser previsto com ganho material sobre baselines simples, usando características do pedido, distância, calendário e histórico da rede vendedor → UF de destino. O alvo binário usa o dia de calendário; o atraso contínuo é tratado de forma descritiva e inferencial. A associação estimada pelo OLS não é interpretada como causalidade.

## Dados e reprodução

A fonte é o *Brazilian E-Commerce Public Dataset by Olist*, publicado no Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Os dados brutos não são versionados. O arquivo brazilian-ecommerce.zip deve ser salvo em data/raw/olist/:

~~~
conda env create -f environment.yml
conda activate ecom-delivery-sla
python scripts/verify_data.py --extract
~~~

Os notebooks são executados na ordem 01 → 11. A versão em português calcula os resultados caros; a versão em inglês carrega os caches correspondentes. A avaliação final abre o teste uma única vez, e o notebook 08 reaproveita os estudos Optuna gravados em SQLite.

~~~
pytest
ruff check src tests scripts
~~~

## Relatório e resultados

O relatório executivo está em site/index.html (português) e site/en.html (inglês). O torneio temporal encontrou CatBoost como único candidato de classificação segundo o gate pré-registrado, mas a avaliação final oficial foi **inconclusiva**. A análise F2 é exploratória e nenhum modelo foi considerado elegível para uso operacional.

## Limitações e documentação

O conjunto Olist cobre um único marketplace e termina em 2018. O período de teste está esgotado para novas avaliações, e production_eligible = false permanece explícito. A documentação completa está em docs/.
