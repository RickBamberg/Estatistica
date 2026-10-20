# dashboard.py
# streamlit run dashboard/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Configurar página
st.set_page_config(
    page_title="Dashboard de Churn - Banco",
    page_icon="📊",
    layout="wide"
)

# Título
st.title("📊 Dashboard de Análise de Churn")
st.markdown("---")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('data/Churn_Modelling.csv')
    return df

df = load_data()

# Sidebar com filtros
st.sidebar.header("🔍 Filtros")
pais_selecionado = st.sidebar.multiselect(
    "País",
    options=df['Geography'].unique(),
    default=df['Geography'].unique()
)

genero_selecionado = st.sidebar.multiselect(
    "Gênero",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

# Aplicar filtros
df_filtrado = df[
    (df['Geography'].isin(pais_selecionado)) &
    (df['Gender'].isin(genero_selecionado))
]

# ---------- MÉTRICAS PRINCIPAIS ----------
st.header("📈 Visão Geral")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_clientes = len(df_filtrado)
    st.metric("Total de Clientes", f"{total_clientes:,}")

with col2:
    total_churn = df_filtrado['Exited'].sum()
    st.metric("Total de Churn", f"{total_churn:,}")

with col3:
    taxa_churn = (total_churn / total_clientes * 100) if total_clientes > 0 else 0
    st.metric("Taxa de Churn", f"{taxa_churn:.1f}%")

with col4:
    clientes_ativos = total_clientes - total_churn
    st.metric("Clientes Ativos", f"{clientes_ativos:,}")

st.markdown("---")

# ---------- GRÁFICOS ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Churn por País")
    churn_pais = df_filtrado.groupby('Geography')['Exited'].mean() * 100
    fig = px.bar(
        churn_pais,
        x=churn_pais.index,
        y=churn_pais.values,
        color=churn_pais.values,
        color_continuous_scale='Reds',
        title="Taxa de Churn por País (%)"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Distribuição de Idade por Churn")
    fig = px.box(
        df_filtrado,
        x='Exited',
        y='Age',
        color='Exited',
        title="Idade vs. Churn",
        labels={'Exited': 'Cancelou?', 'Age': 'Idade'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- SEGUNDA LINHA ----------
col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 Churn por Gênero")
    churn_genero = df_filtrado.groupby('Gender')['Exited'].mean() * 100
    fig = px.pie(
        values=churn_genero.values,
        names=churn_genero.index,
        title="Taxa de Churn por Gênero (%)",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("📊 Atividade vs. Churn")
    churn_atividade = df_filtrado.groupby('IsActiveMember')['Exited'].mean() * 100
    fig = px.bar(
        x=['Inativo', 'Ativo'],
        y=churn_atividade.values,
        color=churn_atividade.values,
        color_continuous_scale='Reds',
        title="Taxa de Churn por Atividade (%)"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- MATRIZ DE CORRELAÇÃO ----------
st.header("📊 Matriz de Correlação")

colunas_correlacao = ['Age', 'Balance', 'NumOfProducts', 'IsActiveMember', 
                     'CreditScore', 'Tenure', 'EstimatedSalary', 'Exited']

corr = df_filtrado[colunas_correlacao].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title="Matriz de Correlação - Fatores de Churn"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- PERFIL DO CLIENTE QUE CANCELA ----------
st.header("📊 Perfil do Cliente que Cancela vs. Fica")

perfil = df_filtrado.groupby('Exited').agg({
    'Age': 'mean',
    'Balance': 'mean',
    'NumOfProducts': 'mean',
    'IsActiveMember': 'mean',
    'CreditScore': 'mean',
    'Tenure': 'mean'
}).round(2)

perfil.index = ['Ficou', 'Cancelou']
st.dataframe(perfil.style.background_gradient(cmap='RdBu_r', axis=1))

st.markdown("---")

# ---------- TOP CLIENTES EM RISCO ----------
st.header("⚠️ Top 10 Clientes com Maior Risco de Churn")

# Calcular score de risco (heurístico)
def calcular_score_risco(row):
    score = 0
    if row['Age'] > 50:
        score += 3
    elif row['Age'] > 40:
        score += 2
    if row['Balance'] > 100000:
        score += 1
    if row['IsActiveMember'] == 0:
        score += 2
    if row['NumOfProducts'] == 1:
        score += 1
    if row['Geography'] == 'Germany':
        score += 2
    return score

df_filtrado['Score_Risco'] = df_filtrado.apply(calcular_score_risco, axis=1)

# Classificar risco
def classificar_risco(score):
    if score >= 7:
        return '🔴 Crítico'
    elif score >= 5:
        return '🟡 Alto'
    elif score >= 3:
        return '🟠 Médio'
    else:
        return '🟢 Baixo'

df_filtrado['Nivel_Risco'] = df_filtrado['Score_Risco'].apply(classificar_risco)

# Top 10 clientes em risco
top_risco = df_filtrado.nlargest(10, 'Score_Risco')[['CustomerId', 'Age', 'Balance', 
                                                    'Geography', 'IsActiveMember', 
                                                    'NumOfProducts', 'Score_Risco', 'Nivel_Risco']]

st.dataframe(top_risco.style.background_gradient(subset=['Score_Risco'], cmap='Reds'))

# ---------- DISTRIBUIÇÃO DE RISCO ----------
st.subheader("📊 Distribuição dos Níveis de Risco")
distribuicao_risco = df_filtrado['Nivel_Risco'].value_counts()
fig = px.pie(
    values=distribuicao_risco.values,
    names=distribuicao_risco.index,
    title="Distribuição dos Clientes por Nível de Risco",
    color_discrete_sequence=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Dashboard desenvolvido para análise de Churn - Dados do Banco")