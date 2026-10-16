# Previsão de Churn em Telecomunicações

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Modelo de classificação para identificar clientes com alta probabilidade de cancelar o serviço (churn), permitindo ações de retenção antes da perda.

## Sumário

- [Contexto](#contexto)
- [Dados](#dados)
- [Metodologia](#metodologia)
- [Visualizações](#visualizações)
- [Resultados](#resultados)
- [Como reproduzir](#como-reproduzir)
- [Próximos passos](#próximos-passos)
- [Fonte dos dados](#fonte-dos-dados)

## Contexto

Em empresas de assinatura, reter um cliente costuma custar menos do que conquistar um novo. O objetivo deste projeto é prever o cancelamento a partir do perfil e do comportamento de contratação do cliente, priorizando a **detecção de quem realmente cancela** (recall da classe positiva) em vez da acurácia geral.

## Dados

Dataset **Telco Customer Churn**, disponibilizado pela IBM. Os dados são fictícios e voltados a fins educacionais.

| Descrição | Valor |
|---|---|
| Clientes (bruto) | 7.043 |
| Variáveis | 21 |
| Clientes após limpeza | 7.032 |
| Taxa de churn | ~26,6% |
| Variável alvo | `Churn` (1 = cancelou, 0 = permaneceu) |

As variáveis incluem perfil do cliente (`SeniorCitizen`, `Partner`, `Dependents`), tempo de relacionamento (`tenure`), serviços contratados, tipo de contrato, forma de pagamento e valores cobrados (`MonthlyCharges`, `TotalCharges`).

## Metodologia

```mermaid
flowchart LR
    A[Dados brutos] --> B[Limpeza]
    B --> C[Separação X e y]
    C --> D[Treino 80% / Teste 20%]
    D --> E[Pré-processamento]
    E --> F[Regressão Logística]
    F --> G[Avaliação no conjunto de teste]
```

1. **Limpeza:** conversão de `TotalCharges` para numérico, remoção de 11 registros com valor ausente e remoção do identificador `customerID`.
2. **Codificação do alvo:** `Churn` mapeado para 0 e 1.
3. **Divisão:** 80% treino e 20% teste, com `stratify` para manter a proporção de churn nos dois conjuntos e `random_state=42` para reprodutibilidade.
4. **Pré-processamento:** `OneHotEncoder` nas variáveis categóricas e `StandardScaler` nas numéricas, encapsulados em um `Pipeline` do scikit-learn para evitar vazamento de dados entre treino e teste.
5. **Modelo base (baseline):** Regressão Logística.

## Visualizações

### Distribuição das classes

<p align="center">
  <img src="images/distribuicao_churn.png" alt="Distribuição de clientes" width="480">
</p>

Aproximadamente 1 em cada 4 clientes cancelou, o que caracteriza um problema com classes desbalanceadas.

### Taxa de churn por tipo de contrato

<p align="center">
  <img src="images/churn_por_contrato.png" alt="Taxa de churn por tipo de contrato" width="480">
</p>

O tipo de contrato mostra grande diferença na taxa de cancelamento, sendo um forte sinal para o modelo.

### Matriz de confusão

<p align="center">
  <img src="images/matriz_confusao.png" alt="Matriz de confusão no conjunto de teste" width="400">
</p>

Mostra quantos clientes o modelo classificou corretamente e onde errou no conjunto de teste.

## Resultados

Avaliação no conjunto de teste (1.407 clientes, sendo 374 com churn).

| Métrica | Classe 0 (permaneceu) | Classe 1 (cancelou) |
|---|---|---|
| Precision | 0,85 | 0,65 |
| Recall | 0,89 | 0,57 |
| F1-score | 0,87 | 0,61 |

**Acurácia geral:** 0,80

**Referência:** um classificador que prevê sempre "permanece" atingiria cerca de 73% de acurácia, portanto o ganho em acurácia é modesto.

**Leitura:** o modelo identifica bem quem permanece, mas detecta apenas 57% dos clientes que cancelam, deixando passar cerca de 43%. Como o custo de perder um cliente tende a superar o custo de uma ação de retenção desnecessária, o recall da classe 1 é a principal métrica a melhorar.

## Como reproduzir

```bash
git clone https://github.com/atararossatti/projeto-churn.git
cd projeto-churn
pip install pandas scikit-learn matplotlib jupyter
jupyter notebook churn.ipynb
```

O notebook baixa o dataset diretamente da fonte, sem necessidade de arquivos locais.

## Próximos passos

- [ ] Ajustar o limiar de decisão para aumentar o recall
- [ ] Comparar com Random Forest e XGBoost/LightGBM
- [ ] Validação cruzada
- [ ] Análise de importância das variáveis
- [ ] Organizar o código em módulos (`src/`) e adicionar `requirements.txt`
- [ ] Servir o modelo via API (FastAPI) com Docker

## Fonte dos dados

IBM, *Telco Customer Churn*.