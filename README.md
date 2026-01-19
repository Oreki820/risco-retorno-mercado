# Risco e retorno no mercado financeiro: o que os dados realmente mostram

Este projeto investiga empiricamente a relação entre risco, retorno e tempo em diferentes classes de ativos, usando apenas dados públicos. Além de análises descritivas (retorno, volatilidade e drawdown), o projeto testa modelos de machine learning para estimar risco futuro (volatilidade), com validação temporal adequada.

O objetivo é educacional e técnico: demonstrar um pipeline reprodutível e conclusões sustentadas por dados, sem recomendações de investimento.

acesse o notebook do projeto com este link: https://colab.research.google.com/drive/1Qm7z3PauLBO71ftCgJChAcvMSEVGId3T?usp=sharing

## Escopo e dados

### Ativos analisados (Yahoo Finance)
- SPY (ações EUA)
- QQQ (tecnologia EUA)
- TLT (treasuries longos)
- GLD (ouro)
- USO (petróleo)
- BTC-USD (cripto)

O recorte adotado usa o período comum entre todos os ativos: **2014-09-17 a 2026-01-16** (datas podem variar conforme atualizações do provedor).

### Variáveis macroeconômicas (FRED)
- FEDFUNDS (taxa básica)
- CPIAUCSL (inflação, índice de preços)
- UNRATE (desemprego)

Os indicadores do World Bank foram testados via API, mas ficaram sem cobertura útil no recorte diário escolhido e, por isso, foram excluídos da modelagem.

## Metodologia

### Métricas principais
- Retornos diários
- Curva de riqueza (retorno acumulado)
- Volatilidade rolling (21 dias, anualizada)
- Drawdown e máximo drawdown
- Correlação contemporânea e rolling

### Machine learning: previsão de risco (não de preço)
O alvo do modelo é a **volatilidade anualizada futura de 21 dias** do SPY, calculada a partir de retornos futuros.

Motivação:
- Volatilidade é um proxy clássico de risco
- É mais estável do que retornos
- Evita o problema comum (e geralmente frágil) de tentar prever preços

Validação:
- Série temporal exige validação temporal (sem embaralhamento)
- Foi usado `TimeSeriesSplit` (walk-forward) para comparação
- Em seguida, foi definido um **holdout final** (últimos 20%) para avaliação fora da amostra

Baselines:
- Baseline forte: `vol_21` (volatilidade recente)
- Baseline mínimo: média do treino

Modelos testados:
- Ridge
- Lasso
- Random Forest

## Resultados principais (resumo)

- A volatilidade apresentou **forte persistência temporal**, tornando o baseline `vol_21` altamente competitivo.
- Modelos mais complexos (Ridge, Lasso, Random Forest), mesmo com variáveis macro e features adicionais, **não superaram consistentemente** o baseline no holdout.
- A principal conclusão é metodológica e empírica: em volatilidade de curto prazo, boa parte do sinal está no próprio histórico recente de risco.

Esses resultados são úteis porque evitam conclusões artificiais e reforçam a necessidade de validação temporal rigorosa.

## Limitações

- O alvo é baseado em volatilidade realizada em janela rolling, o que implica atraso na resposta a choques.
- Séries macro foram alinhadas por forward-fill; não modelam atrasos de divulgação ou revisões.
- Resultados dependem do período analisado e podem variar sob regimes macroeconômicos distintos.
- O projeto é educacional e não contém recomendações de investimento.

## Painel (Streamlit)

O repositório inclui um painel Streamlit (`app.py`) que lê os artifacts gerados no notebook e permite explorar:
- curva de riqueza por ativo
- drawdown
- correlação
- comparação entre observado e previsto para volatilidade futura

## Como rodar localmente

1. Clone o repositório
2. Instale dependências:
   ```bash
   pip install -r requirements.txt
03. Rode:
  ```bash
   streamlit run app.py
  ```
Reprodutibilidade

Os arquivos em artifacts/ foram exportados a partir do notebook (Colab). Eles incluem:

séries processadas (parquet)

modelos treinados (pkl)

métricas em JSON

Autor: Lucas G. F. Gomes


   
