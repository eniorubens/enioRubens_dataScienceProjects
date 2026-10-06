# Publicação no GitHub Pages

A página principal em português é `site/index.html`; a versão em inglês é `site/en.html`. Os gráficos, as sínteses e os dados necessários estão em `site/assets`. O botão de idioma troca entre as versões. O site não depende de caminhos fora da pasta publicada nem de uma CDN para desenhar os gráficos.

O destino é o repositório `eniorubens/enioRubens_dataScienceProjects`, na pasta `CMAPSS_Predictive_Maintenance/`. O endereço do relatório é https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/; a edição inglesa acrescenta `en.html`.

## Conferir e reconstruir

Use um Python com pandas e Plotly para executar `python scripts/build_portfolio_pages.py` na raiz do projeto. O comando utiliza relatórios já salvos e não executa modelagem. Se o README ou o notebook 27 for alterado, reconstrua a página para atualizar as cópias disponíveis para download.

Para conferir localmente, execute `python -m http.server 8765 --directory site --bind 127.0.0.1` e visite `http://127.0.0.1:8765/`. Para encerrar, pressione Ctrl+C no terminal.

## Publicar

No repositório do portfólio, o fluxo ativo fica em `.github/workflows/pages.yml`, na raiz, e usa `CMAPSS_Predictive_Maintenance/site/` e uma página de entrada em `pages-publicacao/`. Em Settings → Pages, selecione GitHub Actions como origem. Na aba Actions, abra Publicar portfólio FD001 e execute Run workflow. O fluxo não copia referências pessoais nem treina modelos. O fluxo dentro da pasta do projeto é apenas um modelo para publicação independente; o GitHub não executa workflows aninhados.

A publicação preserva os projetos já existentes. Os arquivos de código, notebooks, evidências, estudos Optuna e o modelo final ficam no repositório, mas apenas os arquivos estáticos do relatório são enviados ao GitHub Pages. As referências pessoais, caches e os ZIPs duplicados dos dados não são publicados.

## Organização editorial

O README em português é `README.md`; a versão em inglês é `README.en.md`. O notebook 27 possui uma edição canônica em `notebooks/pt-BR` e uma tradução em `notebooks/en-US`. Os demais notebooks preservam a edição canônica em português. A tradução das instruções não altera os nomes das colunas, os comandos ou o comportamento do pipeline.

As cópias do notebook e do README oferecidas no site são arquivos para download. Seus links internos relativos pertencem à estrutura do repositório completo; para seguir esses links, leia os documentos no GitHub ou mantenha a estrutura original após baixar o projeto.

Os gráficos são reconstruídos a partir dos CSVs da análise e conservam as cores acordadas: oportuno em verde no gráfico de alertas; cenário Intermediário em verde no gráfico de custos. O verde do cenário financeiro representa a referência visual e não garante que seja o menor custo em todos os limiares.

Os resultados do benchmark são o registro anterior da comparação. Os alertas, custos e intervalos possuem caráter exploratório de desenvolvimento. A publicação não acrescenta treino, seleção de modelos ou validação independente.
