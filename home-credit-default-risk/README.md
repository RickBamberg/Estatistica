# Análise de Hipóteses — Home Credit Default Risk

Projeto de análise exploratória e teste de hipóteses estatísticas sobre inadimplência de clientes, usando o dataset da competição Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

## 🎯 Objetivo

Investigar quais fatores — demográficos, financeiros, de emprego, de histórico externo de crédito (bureau) e de comportamento de pagamento — estão associados à inadimplência (`TARGET = 1`), testando hipóteses com rigor estatístico (p-valor + tamanho de efeito), complementando com visualizações interativas e insights gerados por LLM.

## 📊 Funcionalidades

### 1. Análise Estatística

- **Testes de hipótese** com p-valor e tamanho de efeito (Cramér's V e r rank-biserial)
- **Tratamento de outliers** e valores sentinela
- **Segmentação de risco** em 4 níveis (Altíssimo, Alto, Médio, Baixo Risco)
- **Métricas financeiras** (perda esperada, exposição por segmento)

### 2. Visualizações

- Gráficos demográficos, financeiros, de emprego, bureau e pagamento
- Matriz de correlação interativa
- Distribuições por status de default
- Perfil detalhado dos segmentos de risco

### 3. Dashboard Interativo (Streamlit)

- **Visão Geral**: KPIs, segmentação de risco, distribuições e correlações
- **Análise por Segmento**: perfil detalhado, variáveis categóricas, estatísticas
- **Simulador de Crédito**: calcule a probabilidade de default para novos clientes
- **Filtros interativos** para explorar diferentes perfis

### 4. Insights com LLM

- Análise automatizada com DeepSeek
- Recomendações acionáveis por segmento
- Insights não óbvios e oportunidades de negócio

## 📁 Estrutura do projeto

```
.
├── dashboard/                           # visualização
│   └── dashboard.py
├── data/                                # arquivos do dataset (não versionados)
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── credit_card_balance.csv
│   ├── installments_payments.csv
│   └── HomeCredit_columns_description.csv
├── hipoteses_home_credit.py             # script principal: tratamento + testes + insights
├── relatorio_hipoteses_home_credit.md   # relatório final consolidado
└── README.md
```


## 🚀 Como rodar

### Instalação das dependências

```bash
pip install pandas numpy scipy requests streamlit plotly matplotlib seaborn scikit-learn
```

**1. Baixar o dataset**

```bash
# Instalar o Kaggle CLI (se ainda não tiver)
pip install kaggle

# Baixar os dados
kaggle competitions download -c home-credit-default-risk

# Descompactar na pasta data/
unzip home-credit-default-risk.zip -d data/
```

**2. Configurar a API Key (opcional - para insights com LLM)**

```bash
export DEEPSEEK_API_KEY="sua_chave_aqui"
```

**3. Executar a análise estatística**

```bash
# Modo interativo (Jupyter)
jupyter notebook hipoteses_home_credit.py

# Ou como script
python hipoteses_home_credit.py
```

**4. Executar o Dashboard Interativo**

```bash
streamlit run dashboard_home_credit.py
```
O dashboard estará disponível em http://localhost:8501

## 📊 Dashboard - Funcionalidades

### 📈 Dashboard (Visão Geral)

* KPIs: Total clientes, defaults, taxa de default, perda esperada

* Segmentação de Risco:

   * Distribuição da base por segmento

   * Taxa de default por segmento

   * Perda esperada por segmento

* Distribuições: Idade, tempo de emprego, EXT_SOURCE_3, crédito/renda

* Matriz de Correlação: Relação entre as principais variáveis

### 📊 Segmentos

* Perfil dos Segmentos: Comparação de idade, tempo de emprego, score externo

* Tabela Detalhada: Clientes, idade, emprego, EXT_SOURCE_3, renda, crédito, default %

* Variáveis Categóricas: Escolaridade, estado civil, tipo de renda, tipo de contrato

* Estatísticas Detalhadas: Média, mediana, desvio padrão por segmento

### 🎯 Simulador

* Entrada de dados: Idade, EXT_SOURCE_3, tempo de emprego, renda, crédito

* Resultados: Probabilidade de default, segmento de risco, recomendação

* Gauge interativo: Visualização do nível de risco

* Detalhamento: Perfil completo do cliente simulado

### 🔍 Principais Descobertas

**Fatores de Risco (Tamanho de Efeito)**

| Variável            | Tamanho do Efeito	| Classificação      |  
|---------------------|-------------------|--------------------|  
| EXT_SOURCE_3	       | -0.3588	         | Moderado           |  
| EXT_SOURCE_1	       | -0.3314	         | Moderado           |  
| EXT_SOURCE_2	       | -0.3122	         | Moderado           |  
| Idade	             | -0.1660	         | Pequeno            |  
| Tempo de Emprego	 | -0.1648	         | Pequeno            |  
| Atraso Médio	       |  0.0853	         | Desprezível        |  
| Escolaridade	       |  0.0576	         | Desprezível        |  
| Tipo de Contrato	 |  0.0309	         | Desprezível        |  
| Razão Crédito/Renda | -0.0042	         | Não significativo  |  

**Segmentação de Risco**

| Segmento	         | Taxa de Default | % da Base    | Perda Esperada  |  
|--------------------|-----------------|--------------|-----------------|  
| Altíssimo Risco    | 19.24%	         | 3.0%	      | R$ 815M         |  
| Alto Risco	      | 12.42%	         | 25.0%	      | R$ 5.58B        |  
| Médio Risco	      | 8.39%	         | 3.3%	      | R$ 1.0B         |  
| Baixo Risco	      | 5.94%	         | 68.7%	      | R$ 7.1B         |  


**Insights Não Óbvios**

1. Razão crédito/renda NÃO é preditiva (p=0.2751) - contrariando o senso comum

2. Número de filhos é praticamente irrelevante (efeito 0.034)

3. Perda concentrada no baixo risco - 51.3% da perda total vem do segmento de baixo risco

4. EXT_SOURCE_3 é o melhor preditor isolado - mais forte que qualquer variável interna

### 📈 Métricas do Modelo

* AUC-ROC: 69.77%

* Acurácia: 91.91%

* Features mais importantes:

   1. EXT_SOURCE_2 (21.5%)
   2. Idade (19.2%)
   3. Atraso Médio (18.5%)
   4. Tempo de Emprego (16.4%)
   5. EXT_SOURCE_3 (14.9%)

### 💡 Recomendações de Negócio

**Para Altíssimo Risco (19.24% default)**

* Suspender novas concessões para EXT_SOURCE_3 < 0.25 e idade < 30

* Reduzir limites em 50%

* Exigir garantias reais

* Prazo máximo de 24 meses

**Para Alto Risco (12.42% default)**

* Aplicar sobretaxa de 300-400 bps

* Limitar prazo a 24 meses

* Monitoramento intensivo

**Para Médio Risco (8.39% default)**

* Manter políticas atuais

* Revisão trimestral

* Cross-sell seletivo

**Para Baixo Risco (5.94% default)**

* Aumentar limites em 15-20% para clientes com EXT_SOURCE_3 > 0.6

* Programa de fidelidade com redução de taxa

* Monitoramento de atraso médio

### 🔧 Configuração Avançada

**Ajuste de Performance**

Para análise completa (307k registros), descomente o nrows no carregamento:

```python
# No dashboard_home_credit.py
df = pd.read_csv("data/application_train.csv")  # Remover nrows para dados completos
```

**Personalização do Dashboard**

* Cores: Ajuste o mapa de cores em color_discrete_map

* KPIs: Adicione ou remova métricas na função plot_kpi_metrics

* Filtros: Expanda os filtros na sidebar adicionando mais opções

### 📝 Licença

Este projeto é para fins educacionais e de análise de dados.

### 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

### 📚 Referências

* **Kaggle - Home Credit Default Risk**

* **Documentação do Streamlit**

* **Plotly Express**

* **DeepSeek API** 

---
