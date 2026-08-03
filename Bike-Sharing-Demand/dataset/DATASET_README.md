# Dataset de bicicletas compartilhadas de Seul com dados meteorológicos

Este diretório contém registros horários agregados de aluguel do sistema
público de bicicletas de Seul, Coreia do Sul, enriquecidos com observações
meteorológicas da Korea Meteorological Administration (KMA).

O arquivo utilizado pelo projeto é `Seoul_public_bicycle.csv`. A cópia
consolidada foi obtida por meio do dataset público
[`lnoahl/seoul-bike-sharing-dataset`](https://www.kaggle.com/datasets/lnoahl/seoul-bike-sharing-dataset)
no Kaggle. Os dados de base permanecem atribuídos às respectivas fontes
governamentais descritas abaixo.

## Descrição

| Coluna | Descrição |
|---|---|
| `ID` | Identificador do registro |
| `datetime` | Data e hora da observação |
| `temperature` | Temperatura do ar em graus Celsius |
| `precipitation` | Precipitação em milímetros |
| `windspeed` | Velocidade do vento em metros por segundo |
| `humidity` | Umidade relativa em percentual |
| `dew_point` | Temperatura do ponto de orvalho |
| `sunshine` | Duração de insolação em horas |
| `solar_radiation` | Radiação solar em MJ/m² |
| `snowfall` | Neve acumulada em centímetros |
| `cloud_cover` | Cobertura de nuvens em oktas |
| `visibility` | Visibilidade em metros |
| `ground_temp` | Temperatura da superfície do solo |
| `weekday` | Dia da semana |
| `holiday` | Indicador de feriado |
| `count` | Total de bicicletas alugadas na hora |

Características principais:

- granularidade horária;
- região de Seul, Coreia do Sul;
- período de setembro de 2015 a dezembro de 2024;
- variáveis de mobilidade e meteorologia integradas.

Os valores ausentes são tratados pelo pipeline do projeto. Sua presença não
deve ser interpretada como autorização para excluir linhas antes da divisão
temporal, pois isso alteraria o contrato de validação.

## Fontes primárias

- [Seoul Open Data Plaza — Public Bike Usage](https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do)
- [Korean Meteorological Administration — ASOS](https://data.kma.go.kr)
- [Cópia consolidada utilizada, publicada no Kaggle](https://www.kaggle.com/datasets/lnoahl/seoul-bike-sharing-dataset)

## Licença e atribuição

Os dados governamentais são disponibilizados sob
[KOGL Type 1 — atribuição obrigatória](https://www.kogl.or.kr/info/license.do).
Eles podem ser utilizados, redistribuídos e adaptados desde que a origem seja
creditada adequadamente.

> 본 저작물은 서울특별시 및 기상청이 공공누리 제1유형으로 개방한 데이터를 바탕으로 가공되었습니다.
>
> 해당 원본 데이터는 공공데이터포털 및 기상자료개방포털에서 자유롭게 내려받을 수 있습니다.

A licença MIT presente na raiz deste projeto cobre somente o código-fonte. Ela
não substitui, remove nem amplia as condições de uso do dataset.

## Aviso

Esta consolidação foi preparada de forma independente para fins educacionais,
analíticos e de pesquisa. O projeto não é afiliado nem endossado pelo Governo
Metropolitano de Seul ou pela Korea Meteorological Administration.
