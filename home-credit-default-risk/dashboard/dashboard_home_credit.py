# dashboard_home_credit_completo.py
# Versão COMPLETA com todos os gráficos, tabelas E simulador funcionando

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Home Credit - Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Estilo CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================
@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados"""
    df = pd.read_csv("data/application_train.csv", nrows=10000)  # Amostra para velocidade
    
    # Tratamento
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
        if col in df.columns:
            df[col] = df[col].abs()
    
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = df["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"] / 365.25
    
    if "AMT_INCOME_TOTAL" in df.columns:
        limite = df["AMT_INCOME_TOTAL"].quantile(0.995)
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=limite)
    
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    
    # Segmentação
    def criar_segmento(row):
        if pd.isna(row['EXT_SOURCE_3']):
            return 'Sem Score'
        elif row['EXT_SOURCE_3'] < 0.3 and row['AGE_YEARS'] < 30:
            return 'Altíssimo Risco'
        elif row['EXT_SOURCE_3'] < 0.45:
            return 'Alto Risco'
        elif row['EXT_SOURCE_3'] < 0.55 and row['EMPLOYED_YEARS'] < 2:
            return 'Médio Risco'
        else:
            return 'Baixo Risco'
    
    df['SEGMENTO_RISCO'] = df.apply(criar_segmento, axis=1)
    df['PERDA_ESPERADA'] = df['TARGET'] * df['AMT_CREDIT']
    
    return df

@st.cache_data
def get_metrics(df):
    """Calcula métricas"""
    total_clientes = len(df)
    total_default = df['TARGET'].sum()
    taxa_default = total_default / total_clientes * 100
    perda_total = df['PERDA_ESPERADA'].sum()
    
    segmentos = df.groupby('SEGMENTO_RISCO', observed=True).agg({
        'TARGET': ['count', 'mean'],
        'PERDA_ESPERADA': 'sum',
        'AMT_CREDIT': ['mean', 'sum']
    }).round(2)
    segmentos.columns = ['count', 'default_rate', 'perda_esperada', 'credito_medio', 'exposicao_total']
    segmentos['default_rate'] = segmentos['default_rate'] * 100
    segmentos['percentual_base'] = segmentos['count'] / total_clientes * 100
    segmentos['percentual_perda'] = segmentos['perda_esperada'] / perda_total * 100
    
    return {
        'total_clientes': total_clientes,
        'total_default': total_default,
        'taxa_default': taxa_default,
        'perda_total': perda_total,
        'segmentos': segmentos
    }

# ============================================================================
# FUNÇÃO DO SIMULADOR
# ============================================================================
def calcula_risco(idade, ext_source, empregado, renda, credito):
    """Calcula risco baseado nos parâmetros"""
    prob_default = 5.0
    
    # Impacto do EXT_SOURCE_3
    if ext_source < 0.2:
        prob_default += 15
    elif ext_source < 0.3:
        prob_default += 10
    elif ext_source < 0.45:
        prob_default += 6
    elif ext_source < 0.6:
        prob_default += 2
    
    # Impacto da idade
    if idade < 25:
        prob_default += 5
    elif idade < 30:
        prob_default += 3
    elif idade > 55:
        prob_default -= 2
    
    # Impacto do tempo de emprego
    if empregado < 1:
        prob_default += 6
    elif empregado < 3:
        prob_default += 3
    elif empregado > 10:
        prob_default -= 2
    
    # Impacto da razão crédito/renda
    if renda > 0:
        razao = credito / (renda * 12)
        if razao > 5:
            prob_default += 4
        elif razao > 3:
            prob_default += 2
    
    prob_default = max(0, min(100, prob_default))
    
    # Determinar segmento
    if ext_source < 0.3 and idade < 30:
        segmento = "Altíssimo Risco"
        cor = "🔴"
    elif ext_source < 0.45:
        segmento = "Alto Risco"
        cor = "🟠"
    elif ext_source < 0.55 and empregado < 2:
        segmento = "Médio Risco"
        cor = "🟡"
    else:
        segmento = "Baixo Risco"
        cor = "🟢"
    
    if prob_default < 5:
        recomendacao = "✅ APROVADO - Baixo risco"
    elif prob_default < 10:
        recomendacao = "ℹ️ ANÁLISE - Risco moderado"
    elif prob_default < 20:
        recomendacao = "⚠️ ANÁLISE DETALHADA - Risco elevado"
    else:
        recomendacao = "❌ RECUSADO - Risco muito alto"
    
    return {
        'probabilidade': prob_default,
        'segmento': segmento,
        'cor': cor,
        'recomendacao': recomendacao,
        'razao_credito_renda': credito / (renda * 12) if renda > 0 else 0
    }

# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO (TODOS OS GRÁFICOS)
# ============================================================================

def plot_kpi_metrics(metricas):
    """KPIs em cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clientes", f"{metricas['total_clientes']:,}")
    with col2:
        st.metric("Total Defaults", f"{metricas['total_default']:,}", delta="⚠️")
    with col3:
        st.metric("Taxa de Default", f"{metricas['taxa_default']:.2f}%")
    with col4:
        st.metric("Perda Esperada", f"R$ {metricas['perda_total']/1e6:.1f}M")

def plot_segmentacao_risco(metricas):
    """Gráficos de segmentação - 3 gráficos lado a lado"""
    df_seg = metricas['segmentos'].reset_index()
    
    # 3 colunas para os gráficos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Pizza - Distribuição da Base
        fig = px.pie(df_seg, values='count', names='SEGMENTO_RISCO',
                     title='Distribuição da Base',
                     color_discrete_sequence=['#ff6b6b', '#ffa94d', '#ffd93d', '#6bcb77'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Barras - Taxa de Default
        colors = ['#dc3545' if x > 10 else '#ffc107' if x > 7 else '#28a745' 
                  for x in df_seg['default_rate']]
        fig = px.bar(df_seg, x='SEGMENTO_RISCO', y='default_rate',
                     title='Taxa de Default',
                     text=df_seg['default_rate'].round(1).astype(str) + '%',
                     color='SEGMENTO_RISCO',
                     color_discrete_map={
                         'Altíssimo Risco': '#dc3545',
                         'Alto Risco': '#ffa94d',
                         'Médio Risco': '#ffd93d',
                         'Baixo Risco': '#6bcb77'
                     })
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False)
        fig.add_hline(y=metricas['taxa_default'], line_dash="dash", 
                      line_color="red", annotation_text="Média Global")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Barras - Perda Esperada
        fig = px.bar(df_seg, x='SEGMENTO_RISCO', y='perda_esperada',
                     title='Perda Esperada (R$)',
                     text=df_seg['perda_esperada'].apply(lambda x: f'R$ {x/1e3:.0f}K'),
                     color='SEGMENTO_RISCO',
                     color_discrete_map={
                         'Altíssimo Risco': '#dc3545',
                         'Alto Risco': '#ffa94d',
                         'Médio Risco': '#ffd93d',
                         'Baixo Risco': '#6bcb77'
                     })
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def plot_distribuicoes(df):
    """Distribuições das principais variáveis"""
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=('Idade', 'Tempo de Emprego', 'EXT_SOURCE_3',
                                        'Crédito/Renda', 'Renda (log)', 'Crédito (log)'))
    
    # Idade
    for target in [0, 1]:
        dados = df[df['TARGET'] == target]['AGE_YEARS'].dropna()
        fig.add_trace(go.Histogram(x=dados, name=f'TARGET={target}', opacity=0.6,
                                   marker_color='#1f77b4' if target==0 else '#ff7f0e'),
                      row=1, col=1)
    
    # Tempo de Emprego
    for target in [0, 1]:
        dados = df[df['TARGET'] == target]['EMPLOYED_YEARS'].dropna()
        fig.add_trace(go.Histogram(x=dados, name=f'TARGET={target}', opacity=0.6,
                                   marker_color='#1f77b4' if target==0 else '#ff7f0e',
                                   showlegend=False),
                      row=1, col=2)
    
    # EXT_SOURCE_3
    for target in [0, 1]:
        dados = df[df['TARGET'] == target]['EXT_SOURCE_3'].dropna()
        fig.add_trace(go.Histogram(x=dados, name=f'TARGET={target}', opacity=0.6,
                                   marker_color='#1f77b4' if target==0 else '#ff7f0e',
                                   showlegend=False),
                      row=1, col=3)
    
    # Crédito/Renda
    for target in [0, 1]:
        dados = df[df['TARGET'] == target]['CREDIT_INCOME_RATIO'].dropna()
        fig.add_trace(go.Histogram(x=dados, name=f'TARGET={target}', opacity=0.6,
                                   marker_color='#1f77b4' if target==0 else '#ff7f0e',
                                   showlegend=False),
                      row=2, col=1)
    
    # Renda (log)
    for target in [0, 1]:
        dados = np.log1p(df[df['TARGET'] == target]['AMT_INCOME_TOTAL'].dropna())
        fig.add_trace(go.Histogram(x=dados, name=f'TARGET={target}', opacity=0.6,
                                   marker_color='#1f77b4' if target==0 else '#ff7f0e',
                                   showlegend=False),
                      row=2, col=2)
    
    # Crédito (log)
    for target in [0, 1]:
        dados = np.log1p(df[df['TARGET'] == target]['AMT_CREDIT'].dropna())
        fig.add_trace(go.Histogram(x=dados, name=f'TARGET={target}', opacity=0.6,
                                   marker_color='#1f77b4' if target==0 else '#ff7f0e',
                                   showlegend=False),
                      row=2, col=3)
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Idade (anos)", row=1, col=1)
    fig.update_xaxes(title_text="Tempo de Emprego (anos)", row=1, col=2)
    fig.update_xaxes(title_text="EXT_SOURCE_3", row=1, col=3)
    fig.update_xaxes(title_text="Crédito/Renda", row=2, col=1)
    fig.update_xaxes(title_text="Log(Renda)", row=2, col=2)
    fig.update_xaxes(title_text="Log(Crédito)", row=2, col=3)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_analise_categorica(df):
    """Análise de variáveis categóricas"""
    colunas_cat = ['NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
                   'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE']
    
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=colunas_cat)
    
    for i, col in enumerate(colunas_cat):
        row = i//2 + 1
        col_pos = i%2 + 1
        
        if col in df.columns:
            df_group = df.groupby(col, observed=True)['TARGET'].mean() * 100
            df_group = df_group.sort_values()
            
            fig.add_trace(
                go.Bar(x=df_group.values, y=df_group.index, orientation='h',
                       text=df_group.values.round(1).astype(str) + '%',
                       textposition='outside',
                       marker_color=px.colors.qualitative.Set3),
                row=row, col=col_pos
            )
            
            fig.add_vline(x=df['TARGET'].mean()*100, line_dash="dash", 
                          line_color="red", row=row, col=col_pos)
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Taxa de Default (%)")
    st.plotly_chart(fig, use_container_width=True)

def plot_perfil_segmentos(df):
    """Perfil detalhado dos segmentos"""
    segmentos = ['Altíssimo Risco', 'Alto Risco', 'Médio Risco', 'Baixo Risco']
    
    perfis = []
    for seg in segmentos:
        dados = df[df['SEGMENTO_RISCO'] == seg]
        if len(dados) > 0:
            perfis.append({
                'Segmento': seg,
                'Clientes': len(dados),
                'Idade': dados['AGE_YEARS'].mean(),
                'Tempo Emprego': dados['EMPLOYED_YEARS'].mean(),
                'EXT_SOURCE_3': dados['EXT_SOURCE_3'].mean(),
                'Renda': dados['AMT_INCOME_TOTAL'].mean(),
                'Crédito': dados['AMT_CREDIT'].mean(),
                'Default %': dados['TARGET'].mean() * 100
            })
    
    df_perfil = pd.DataFrame(perfis)
    
    # Gráfico de barras agrupadas
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_perfil['Segmento'], y=df_perfil['Idade'], 
                         name='Idade', marker_color='blue'))
    fig.add_trace(go.Bar(x=df_perfil['Segmento'], y=df_perfil['Tempo Emprego'], 
                         name='Tempo Emprego', marker_color='orange'))
    fig.add_trace(go.Bar(x=df_perfil['Segmento'], y=df_perfil['EXT_SOURCE_3'] * 100, 
                         name='EXT_SOURCE_3 (*100)', marker_color='green'))
    fig.add_trace(go.Bar(x=df_perfil['Segmento'], y=df_perfil['Default %'], 
                         name='Default %', marker_color='red'))
    
    fig.update_layout(title='Perfil dos Segmentos de Risco',
                      barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela completa
    st.markdown("### 📋 Tabela Detalhada por Segmento")
    st.dataframe(df_perfil.style.format({
        'Idade': '{:.1f}',
        'Tempo Emprego': '{:.1f}',
        'EXT_SOURCE_3': '{:.2f}',
        'Renda': 'R$ {:.0f}',
        'Crédito': 'R$ {:.0f}',
        'Default %': '{:.2f}%'
    }), use_container_width=True)

def plot_matriz_correlacao(df):
    """Matriz de correlação"""
    cols = ['AGE_YEARS', 'EMPLOYED_YEARS', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 
            'EXT_SOURCE_3', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CREDIT_INCOME_RATIO', 'TARGET']
    
    cols_exist = [c for c in cols if c in df.columns]
    df_corr = df[cols_exist].corr()
    
    fig = px.imshow(df_corr, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Matriz de Correlação')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_simulador(df):
    """Simulador interativo - Versão completa"""
    st.subheader("🎯 Simulador de Aprovação de Crédito")
    st.markdown("Ajuste os parâmetros abaixo para simular a aprovação de um cliente")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Dados do Cliente")
        
        idade = st.number_input("Idade (anos)", min_value=18, max_value=80, value=35, step=1)
        ext_source = st.slider("EXT_SOURCE_3 (Score Externo)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
        empregado = st.number_input("Tempo de Emprego (anos)", min_value=0, max_value=40, value=5, step=1)
        renda = st.number_input("Renda Mensal (R$)", min_value=1000, max_value=50000, value=5000, step=500)
        credito = st.number_input("Crédito Solicitado (R$)", min_value=1000, max_value=200000, value=10000, step=1000)
    
    resultado = calcula_risco(idade, ext_source, empregado, renda, credito)
    
    with col2:
        st.markdown("### 📊 Resultado da Simulação")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Probabilidade de Default", f"{resultado['probabilidade']:.1f}%")
        with col_b:
            st.metric("Segmento", f"{resultado['cor']} {resultado['segmento']}")
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=resultado['probabilidade'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgreen"},
                    {'range': [5, 10], 'color': "yellowgreen"},
                    {'range': [10, 20], 'color': "orange"},
                    {'range': [20, 100], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recomendação
        cor_fundo = '#d4edda' if 'APROVADO' in resultado['recomendacao'] else '#fff3cd' if 'ANÁLISE' in resultado['recomendacao'] else '#f8d7da'
        cor_borda = '#28a745' if 'APROVADO' in resultado['recomendacao'] else '#ffc107' if 'ANÁLISE' in resultado['recomendacao'] else '#dc3545'
        st.markdown(f"""
        <div style="background-color:{cor_fundo}; padding:15px; border-radius:10px; 
                    border:2px solid {cor_borda}; text-align:center; font-size:18px; font-weight:bold;">
            {resultado['recomendacao']}
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📈 Detalhamento do Perfil"):
            st.write(f"""
            - **Idade**: {idade} anos
            - **EXT_SOURCE_3**: {ext_source:.2f}
            - **Tempo de Emprego**: {empregado} anos
            - **Renda Mensal**: R$ {renda:,.2f}
            - **Crédito Solicitado**: R$ {credito:,.2f}
            - **Razão Crédito/Renda**: {resultado['razao_credito_renda']:.1f}x
            """)

# ============================================================================
# MAIN
# ============================================================================
def main():
    st.markdown('<div class="main-header">🏦 Home Credit - Análise de Risco de Crédito</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Carregar dados
    with st.spinner("🔄 Carregando dados..."):
        df = load_and_process_data()
        metricas = get_metrics(df)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
        st.markdown("## 📊 Menu")
        
        opcao = st.radio(
            "Selecione uma seção:",
            ["📈 Dashboard", "📊 Segmentos", "📉 Análise Detalhada", "🎯 Simulador"]
        )
        
        st.markdown("---")
        st.markdown(f"**Total:** {metricas['total_clientes']:,} clientes")
        st.markdown(f"**Default:** {metricas['taxa_default']:.2f}%")
    
    # Conteúdo
    if opcao == "📈 Dashboard":
        st.markdown("## 📊 Visão Geral do Portfólio")
        plot_kpi_metrics(metricas)
        st.markdown("---")
        
        st.markdown("### 📈 Segmentação de Risco")
        plot_segmentacao_risco(metricas)
        
        st.markdown("### 📊 Distribuições das Principais Variáveis")
        plot_distribuicoes(df)
        
        st.markdown("### 🔗 Matriz de Correlação")
        plot_matriz_correlacao(df)
    
    elif opcao == "📊 Segmentos":
        st.markdown("## 📊 Análise por Segmento")
        plot_perfil_segmentos(df)
        
        st.markdown("---")
        st.markdown("### 📈 Variáveis Categóricas")
        plot_analise_categorica(df)
        
        # Tabela de estatísticas detalhadas
        st.markdown("### 📋 Estatísticas Detalhadas")
        stats = df.groupby('SEGMENTO_RISCO', observed=True).agg({
            'TARGET': ['count', 'mean'],
            'AGE_YEARS': ['mean', 'median', 'std'],
            'EMPLOYED_YEARS': ['mean', 'median', 'std'],
            'EXT_SOURCE_3': ['mean', 'median', 'std'],
            'AMT_INCOME_TOTAL': ['mean', 'median'],
            'AMT_CREDIT': ['mean', 'median']
        }).round(2)
        st.dataframe(stats, use_container_width=True)
    
    elif opcao == "📉 Análise Detalhada":
        st.markdown("## 📉 Análise Detalhada")
        
        tab1, tab2, tab3 = st.tabs(["📊 Distribuições", "📈 Categóricas", "🔗 Correlação"])
        
        with tab1:
            plot_distribuicoes(df)
        
        with tab2:
            plot_analise_categorica(df)
        
        with tab3:
            plot_matriz_correlacao(df)
    
    elif opcao == "🎯 Simulador":
        plot_simulador(df)
    
    # Footer
    st.markdown("---")
    st.caption("📊 Dashboard interativo - Home Credit Default Risk Analysis")

if __name__ == "__main__":
    main()