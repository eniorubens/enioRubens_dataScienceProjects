# Proveniência — CVRPLIB, conjunto Solomon (100 clientes)

- Fonte: CVRPLIB (PUC-Rio), https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances — "VRPTW Set Solomon (Solomon, 1987)"
- Baixado em: 2026-11-17
- Solomon.7z  <- .../en/download/instance-set/22      sha256 55ba67b56a47855432f1b80164db7cc87ce3c7c22bc9855f2cc0013c81d92a37
- Solomon-solutions.7z <- .../en/download/instance-set-bks/22 sha256 88a5d385c0faf63e8eebc6a1bc755ced85d92575b816d33737b0258df459394e
- Convenção declarada pela fonte: objetivo = minimizar distância total; distância euclidiana truncada em 1 casa decimal;
  custo arredondado a 1 casa; número de rotas <= K (K=25). NÃO é a convenção hierárquica do SINTEF (veículos, depois distância).
- Licença: nenhuma licença explícita encontrada na página; citar Solomon (1987), Operations Research 35(2), e o CVRPLIB.
- extracted/ é derivado dos .7z (descompactado com tar.exe do Windows/libarchive).
- Derivado versionado: data/reference/cvrplib_bks_100.csv (instance, routes em JSON, cost com 1 casa, source=CVRPLIB), gerado por scripts/build_cvrplib_reference.py a partir de extracted/bks/*.sol (pasta extracted/ não é versionada).
