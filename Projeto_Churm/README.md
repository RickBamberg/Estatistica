# 🏦 Análise de Churn em Banco

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-API-purple.svg)](https://deepseek.com/)

## 📋 Visão Geral

Este projeto tem como objetivo **analisar e prever o churn (cancelamento) de clientes** em uma instituição bancária. Utilizando uma abordagem que combina **teste de hipóteses**, **heurísticas**, **Machine Learning** e **Inteligência Artificial (DeepSeek)**, o projeto identifica os principais fatores que levam ao cancelamento e fornece recomendações acionáveis para retenção de clientes.

### 🎯 Objetivos

- **Validar estatisticamente** quais fatores influenciam o churn
- **Criar heurísticas** (regras práticas) para identificar clientes em risco
- **Construir um modelo preditivo** para estimar a probabilidade de churn
- **Desenvolver um dashboard interativo** para visualização dos insights
- **Integrar um LLM (DeepSeek)** para geração automática de insights e recomendações

---

## 🤖 Integração com DeepSeek (LLM)

O projeto utiliza a API da **DeepSeek** para gerar automaticamente análises aprofundadas, insights não óbvios e recomendações de negócio baseadas nos dados.

### O que a IA faz:

| Funcionalidade                | Descrição                                                     |  
|-------------------------------|---------------------------------------------------------------|  
| **📋 Resumo Executivo**       | Gera um resumo em linguagem simples dos principais achados    |  
| **💡 Insights Principais**    | Identifica padrões e tendências com números concretos         |  
| **🎯 Recomendações**          | Sugere ações específicas para cada perfil de risco            |  
| **🔍 Insights Não Óbvios**    | Revela padrões ocultos que não estão na superfície            |  
| **💬 Perguntas Interativas**  | Permite fazer perguntas sobre os dados em linguagem natural   |  

### Exemplo de Insight Gerado pela IA:

> *"O banco está perdendo seus clientes mais valiosos. Clientes com maior saldo são os que mais saem (R$ 91 mil vs R$ 72 mil). Isso sugere que o banco não está oferecendo valor proporcional ao patrimônio do cliente."*

---

## 📊 Principais Descobertas

### O Perfil do Cliente que Cancela

| Característica    | Cliente que Fica  | Cliente que Cancela   | Impacto                                   |  
| :---              | :---              | :---                  | :---                                      |  
| **Idade**         | 37.4 anos         | **44.8 anos**         | 🔴 Clientes mais velhos cancelam mais     |  
| **Saldo**         | R$ 72.745         | **R$ 91.109**         | 🔴 Clientes mais ricos são mais exigentes |  
| **Atividade**     | 55% ativos        | **36% ativos**        | 🔴 Inatividade é um alerta vermelho       |  
| **Produtos**      | 1.54 produtos     | **1.48 produtos**     | 🟡 Menos produtos = menos laços           |  
| **País**          | 16.2% (França)    | **32.4% (Alemanha)**  | 🔴 Alemanha tem o dobro de churn          |  

### Score de Risco

| Score | Taxa de Churn | Nível de Risco    |  
| :---  | :---          | :---              |  
| 0-2   | < 10%         | 🟢 Baixo          |  
| 3-4   | ~20%          | 🟠 Médio          |  
| 5-6   | ~50%          | 🔴 Alto           |  
| 7+    | **~85%**      | 🔴🔴 Crítico      | 

---

## 🧠 Metodologia

### 1. Teste de Hipóteses

Utilizei o **teste t de Student** para validar estatisticamente quais variáveis influenciam o churn:

```python
from scipy import stats

# Exemplo: Testando se idade influencia churn
churn = df[df['Exited'] == 1]['Age']
nao_churn = df[df['Exited'] == 0]['Age']

t_stat, p_valor = stats.ttest_ind(churn, nao_churn)
# p-valor < 0.001 → diferença é estatisticamente significativa!
```

**Hipóteses Validadas:**

* ✅ Idade - Clientes mais velhos cancelam mais (p < 0.001)

* ✅ Saldo - Clientes com maior saldo cancelam mais (p < 0.001)

* ✅ Atividade - Clientes inativos têm maior churn (p < 0.001)

* ✅ País - Alemanha tem maior taxa de churn (p < 0.001)

* ❌ Tempo de Casa - Não influencia (p = 0.16)

* ❌ Posse de Cartão - Não influencia (p = 0.49)

### 2. Heurísticas (Score de Risco)

Criei um score de risco baseado em regras práticas:

```python
def calcular_score_risco(row):
    score = 0
    if row['Age'] > 50: score += 3
    elif row['Age'] > 40: score += 2
    if row['Balance'] > 100000: score += 1
    if row['IsActiveMember'] == 0: score += 2
    if row['NumOfProducts'] == 1: score += 1
    if row['Geography'] == 'Germany': score += 2
    return score
```

### 3. Machine Learning

**Modelo:** Random Forest com SMOTE para balanceamento

Métrica	            | Valor |  
--------------------|-------|  
Recall (Churn)	    | 65%   | 
Precisão (Churn)    | 52%   |  
F1-Score (Churn)    | 0.58  |  
Acurácia	        | 81%   |  

**Features Utilizadas:**

* Age, Balance, IsActiveMember, Geography, NumOfProducts, CreditScore

**Features Excluídas (por razões éticas ou baixa relevância):**

* Gender - Discriminatório

* Tenure, HasCrCard, EstimatedSalary - Baixo impacto preditivo

### 4. Integração com LLM (DeepSeek)

O projeto utiliza a API da DeepSeek para gerar insights automaticamente:

```python
class DeepSeekClient:
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com/chat/completions"
    
    def chat(self, messages, temperature=0.7, max_tokens=2000):
        """Envia uma mensagem para a DeepSeek"""
        # ... código completo no dashboard
```

---

## 📂 Estrutura do Projeto

```text
projeto_churn/
│
├── data/
│   └── Churn_Modelling.csv          # Dataset do Kaggle
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb   # EDA e estatísticas
│   ├── 02_teste_hipoteses.ipynb        # Teste t e validações
│   ├── 03_modelo_ml.ipynb              # Random Forest + SMOTE
│   └── 04_dashboard.ipynb              # Dashboard Plotly
│
├── dashboard/
│   ├── dashboard.py                    # Dashboard interativo
│   └── dashboard_ia.py                 # Dashboard interativo com IA
│
├── outputs/
│   ├── dashboard_streamlit.png         # Screenshot do dashboard
│   └── modelo_final.pkl                # Modelo treinado (opcional)
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 Como Executar o Projeto

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/projeto-churn.git
cd projeto-churn
```

### 2. Configure a Chave da DeepSeek

**Windows (PowerShell):**

```bash
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sua-chave-aqui", "User")
```

**Linux/Mac:**

```bash
export DEEPSEEK_API_KEY="sua-chave-aqui"
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o Dashboard Interativo

**Opção A: Streamlit (Recomendado)**

```bash
streamlit run dashboard/streamlit_app.py
```

**Opção B: Jupyter Notebook**

```bash
jupyter notebook notebooks/04_dashboard.ipynb
```

### 5. Treine o Modelo (Opcional)

```bash
# Execute o notebook de modelo
jupyter notebook notebooks/03_modelo_ml.ipynb
```

---

## 📊 Dashboard Interativo com IA

**O dashboard permite visualizar:**

* 📈 Métricas principais (total de clientes, taxa de churn)

* 🌍 Churn por país (com destaque para Alemanha)

* 👤 Distribuição de idade e fatores de risco

* ⚠️ Top 10 clientes com maior probabilidade de churn

* 🔍 Filtros por país e gênero

**Funcionalidades:**

**Aba	Funcionalidade**
| Aba                           | Descrição                                                         |
|-------------------------------|-------------------------------------------------------------------| 
| 📊 Visualizações	            | Gráficos interativos: churn por país, idade, atividade, saldo     |  
| 🤖 Insights com DeepSeek      | Geração automática de insights e perguntas em linguagem natural   |  
| 📝 Relatório Completo	        | Relatório detalhado com todas as métricas e recomendações         |  


**Exemplo de Perguntas para a IA:**

* "Por que clientes da Alemanha cancelam mais?"

* "Qual a idade ideal para campanhas de retenção?"

* "O que fazer para reduzir o churn em 5%?"

* "Qual o perfil do cliente que nunca cancela?"


**Screenshot do Dashboard**

https://outputs/dashboard_streamlit.png

## 💡 Recomendações de Negócio

| Grupo de Risco            | Estratégia de Retenção                                    |    
|---------------------------|-----------------------------------------------------------|    
| 50+ anos com saldo alto	| Atendimento personalizado, consultoria financeira         |    
| Inativos	                | Campanha de reengajamento ("Volte e ganhe benefícios")    |    
| 1 produto	                | Cross-selling: oferecer cartão, investimentos, seguros    |    
| Score ≥ 5	                | Contato proativo por telefone (não apenas email)          |    
| Alemanha	                | Estratégias específicas para o mercado alemão             |  

## Insights Não Óbvios (Identificados pela IA)

1. O paradoxo do saldo: Clientes com mais dinheiro são os que mais saem - o banco não está oferecendo valor
   proporcional ao patrimônio.

2. Idade como ciclo de vida: Clientes na faixa dos 44,8 anos estão em fase de mudanças (casamento, filhos, imóvel)
   o banco pode antecipar essas necessidades.

3. Inatividade como sintoma: Queda de transações é um alerta precoce melhor que esperar o cliente ficar totalmente
   inativo.

4. Alemanha como problema estrutural: Churn de 32,4% sugere problemas de precificação ou concorrência local.

---

## 🔧 Tecnologias Utilizadas

| Categoria	        | Tecnologias                               |  
|-------------------|-------------------------------------------|  
| Linguagem	        | Python 3.8+                               |  
| Análise de Dados	| Pandas, NumPy, SciPy                      |  
| Visualização	    | Matplotlib, Seaborn, Plotly               |  
| Machine Learning  | Scikit-learn, Imbalanced-learn (SMOTE)    |  
| Dashboard	        | Streamlit, Plotly                         |  
| Estatística	    | Teste t, Qui-Quadrado                     |  
| IA/LLM	        | DeepSeek API                              |  

---

## 📈 Resultados e Métricas

**Modelo Final**

| Classe        | Precisão  | Recall    | F1-Score  |  
|---------------|-----------|-----------|-----------|  
| Não Churn (0)	| 91%	    | 85%       | 0.88      |  
| Churn (1)	    | 52%	    |65%	    | 0.58      |  

**Score de Risco**

| Score | Churn Rate    | Interpretação |  
|-------|---------------|---------------|  
| 0	    | 2.7%	        | Risco mínimo  |  
| 3	    | 20.0%	        | Atenção       |  
| 5	    | 44.1%	        | Alto risco    |  
| 7	    | 85.7%	        | Crítico       |  

---

## 🎓 Principais Aprendizados

1. Teste de Hipóteses - Como validar estatisticamente fatores de risco

2. Heurísticas - Como criar regras práticas a partir de dados

3. Balanceamento de Dados - Como usar SMOTE para lidar com dados desbalanceados

4. Interpretabilidade - Como explicar modelos para stakeholders

5. Ética em Dados - Como evitar variáveis discriminatórias (gênero)

6. Integração com LLM - Como usar IA para gerar insights automáticos

---

## 👥 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

1. Fork o projeto

2. Crie sua branch (git checkout -b feature/AmazingFeature)

3. Commit suas mudanças (git commit -m 'Add some AmazingFeature')

4. Push para a branch (git push origin feature/AmazingFeature)

5. Abra um Pull Request

---

## 📝 Licença

Distribuído sob a licença MIT. Veja LICENSE para mais informações.

---

## 📧 Contato

Nome: **Carlos Henrique Bamberg Marques**

E-mail: rick.bamber@gmail.com

LinkedIn: https://www.linkedin.com/in/carlos-henrique-bamberg-marques

GitHub: https://github.com/RickBamberg/Portfolio/

---

## 🙏 Agradecimentos

**Kaggle** pelo dataset

**Streamlit** pelo framework de dashboard

**Comunidade de Ciência de Dados** pelos aprendizados

**DeepSeek** pela API de IA

---

## 📚 Referências

* Bank Customer Churn Dataset (Kaggle)

* Scikit-learn Documentation

* Plotly Documentation

* DeepSeek API Documentation
