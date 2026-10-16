
# Projeto Churn: prevendo cancelamento de clientes

## O problema
Uma empresa de telefone e internet quer descobrir quais clientes vão cancelar o serviço, para agir antes de perdê-los.

## Os dados
Dataset Telco Customer Churn, disponibilizado pela IBM (dados fictícios para estudo). São 7043 clientes e 21 colunas.

## O que foi feito
- Limpeza dos dados (coluna TotalCharges convertida para número, linhas vazias removidas)
- Divisão em 80% para treino e 20% para teste
- Primeiro modelo: Regressão Logística

## Resultados do primeiro modelo
- Acurácia: 80%
- Recall dos clientes que cancelaram: 57%
- Precision dos clientes que cancelaram: 65%

O modelo acerta bem quem fica, mas ainda deixa passar boa parte de quem cancela.

## Próximos passos
- Testar modelos melhores (Random Forest, XGBoost)
- Melhorar o recall
- Criar uma API para usar o modelo