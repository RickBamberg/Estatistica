# dash.py - Versão corrigida com set_page_config no topo

import streamlit as st

# ==================== CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA COISA!) ====================
st.set_page_config(
    page_title="Dashboard de Churn com DeepSeek",
    page_icon="🤖",
    layout="wide"
)

# ==================== AGORA SIM, O RESTO DO CÓDIGO ====================
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import openai

# ==================== CONFIGURAÇÃO DO DEEPSEEK (VERSÃO 3.3.1) ====================
# Para openai 3.3.1 - usa a sintaxe global
openai.api_key = os.environ.get('DEEPSEEK_API_KEY')
openai.api_base = "https://api.deepseek.com/v1"

# Verificar se a chave está configurada
if not openai.api_key:
    st.sidebar.error("❌ Chave DEEPSEEK_API_KEY não encontrada!")
else:
    st.sidebar.success("✅ DeepSeek configurado!")

# ==================== CARREGAR DADOS ====================
@st.cache_data
def load_data():
    df = pd.read_csv('data/Churn_Modelling.csv')
    return df

df = load_data()

# ==================== CALCULAR RESULTADOS ====================
@st.cache_data
def calcular_resultados(df):
    from scipy import stats
    
    # Testes de hipóteses
    churn = df[df['Exited'] == 1]['Age']
    nao_churn = df[df['Exited'] == 0]['Age']
    t_stat, p_idade = stats.ttest_ind(churn, nao_churn)
    
    churn = df[df['Exited'] == 1]['Balance']
    nao_churn = df[df['Exited'] == 0]['Balance']
    t_stat, p_balance = stats.ttest_ind(churn, nao_churn)
    
    churn = df[df['Exited'] == 1]['IsActiveMember']
    nao_churn = df[df['Exited'] == 0]['IsActiveMember']
    t_stat, p_active = stats.ttest_ind(churn, nao_churn)
    
    churn = df[df['Exited'] == 1]['CreditScore']
    nao_churn = df[df['Exited'] == 0]['CreditScore']
    t_stat, p_credit = stats.ttest_ind(churn, nao_churn)
    
    feature_importance = {
        'Idade': 0.28,
        'Saldo': 0.22,
        'Atividade': 0.20,
        'País (Alemanha)': 0.15,
        'Nº Produtos': 0.10,
        'Score de Crédito': 0.05
    }
    
    return {
        'p_idade': p_idade,
        'p_balance': p_balance,
        'p_active': p_active,
        'p_credit': p_credit,
        'feature_importance': feature_importance,
        'taxa_churn': df['Exited'].mean() * 100,
        'total_clientes': len(df),
        'total_churn': df['Exited'].sum()
    }

resultados = calcular_resultados(df)

# ==================== FUNÇÃO PARA GERAR INSIGHTS (VERSÃO 3.3.1) ====================
def gerar_insights_deepseek(df_filtrado, resultados):
    """Gera insights usando a API da DeepSeek - compatível com openai 3.3.1"""
    
    # Preparar dados para o prompt
    prompt = f"""
Você é um especialista em análise de clientes (Customer Analytics) e retenção.

Analise os dados de churn de um banco e gere insights acionáveis.

DADOS GERAIS:
- Total de clientes: {len(df_filtrado)}
- Clientes que cancelaram: {df_filtrado['Exited'].sum()}
- Taxa de churn: {resultados['taxa_churn']:.1f}%

PERFIL DO CLIENTE QUE CANCELA:
- Idade média de quem cancela: {df_filtrado[df_filtrado['Exited']==1]['Age'].mean():.1f} anos
- Idade média de quem fica: {df_filtrado[df_filtrado['Exited']==0]['Age'].mean():.1f} anos
- Saldo médio de quem cancela: R$ {df_filtrado[df_filtrado['Exited']==1]['Balance'].mean():.2f}
- Saldo médio de quem fica: R$ {df_filtrado[df_filtrado['Exited']==0]['Balance'].mean():.2f}
- % ativos entre quem cancela: {df_filtrado[df_filtrado['Exited']==1]['IsActiveMember'].mean()*100:.1f}%
- % ativos entre quem fica: {df_filtrado[df_filtrado['Exited']==0]['IsActiveMember'].mean()*100:.1f}%
- Média de produtos de quem cancela: {df_filtrado[df_filtrado['Exited']==1]['NumOfProducts'].mean():.2f}
- Média de produtos de quem fica: {df_filtrado[df_filtrado['Exited']==0]['NumOfProducts'].mean():.2f}

CHURN POR PAÍS:
{df_filtrado.groupby('Geography')['Exited'].mean().to_string()}

TESTES DE HIPÓTESE (p-valor):
- Idade: {resultados['p_idade']:.5f} {'✅ significativo' if resultados['p_idade'] < 0.05 else '❌ não significativo'}
- Saldo: {resultados['p_balance']:.5f} {'✅ significativo' if resultados['p_balance'] < 0.05 else '❌ não significativo'}
- Atividade: {resultados['p_active']:.5f} {'✅ significativo' if resultados['p_active'] < 0.05 else '❌ não significativo'}

FEATURES MAIS IMPORTANTES:
{json.dumps(resultados['feature_importance'], indent=2, ensure_ascii=False)}

Com base nesses dados, forneça:

1. Resumo Executivo (2-3 parágrafos em linguagem simples)

2. Principais Insights (4-5 bullet points com números concretos)

3. Recomendações de Ação (estratégias práticas para cada perfil de risco)

4. Insights Não Óbvios (padrões interessantes que não estão na superfície)

Seja objetivo, prático, use números e evite jargões técnicos desnecessários.
Responda em português.
"""
    
    try:
        # Para openai 3.3.1
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de churn e retenção de clientes. Responda sempre em português."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro ao gerar insights: {str(e)}"

# ==================== INTERFACE STREAMLIT ====================

# Sidebar
st.sidebar.title("⚙️ Configurações")
st.sidebar.info(f"""
**🤖 DeepSeek API**
- Modelo: DeepSeek-V3
- Status: {'✅ Conectado' if openai.api_key else '❌ Não conectado'}
""")

# Filtros
st.sidebar.title("🔍 Filtros")
paises = st.sidebar.multiselect("País", options=['France', 'Spain', 'Germany'], default=['France', 'Spain', 'Germany'])
generos = st.sidebar.multiselect("Gênero", options=['Female', 'Male'], default=['Female', 'Male'])

# Aplicar filtros
df_filtrado = df[(df['Geography'].isin(paises)) & (df['Gender'].isin(generos))]

# Título
st.title("🏦 Dashboard de Análise de Churn")
st.caption("Análise completa com insights gerados por DeepSeek AI 🤖")

# Métricas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Clientes", f"{len(df_filtrado):,}")
with col2:
    st.metric("Total de Churn", f"{df_filtrado['Exited'].sum():,}")
with col3:
    taxa = (df_filtrado['Exited'].sum() / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
    st.metric("Taxa de Churn", f"{taxa:.1f}%")
with col4:
    ativos = len(df_filtrado) - df_filtrado['Exited'].sum()
    st.metric("Clientes Ativos", f"{ativos:,}")

st.markdown("---")

# Abas
tab1, tab2, tab3 = st.tabs(["📊 Visualizações", "🤖 Insights com DeepSeek", "📝 Relatório Completo"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        churn_pais = df_filtrado.groupby('Geography')['Exited'].mean() * 100
        fig = px.bar(x=churn_pais.index, y=churn_pais.values, color=churn_pais.values, color_continuous_scale='Reds', title="Taxa de Churn por País (%)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df_filtrado, x='Exited', y='Age', color='Exited', title="Distribuição de Idade", color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        churn_atividade = df_filtrado.groupby('IsActiveMember')['Exited'].mean() * 100
        fig = px.bar(x=['Inativo', 'Ativo'], y=churn_atividade.values, color=['Inativo', 'Ativo'], color_discrete_sequence=['#e74c3c', '#2ecc71'], title="Taxa de Churn por Atividade (%)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(df_filtrado, x='Exited', y='Balance', color='Exited', title="Distribuição de Saldo", color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🤖 Insights Gerados por DeepSeek")
    
    if not openai.api_key:
        st.error("""
❌ DeepSeek não conectado!

Verifique:
1. A variavel de ambiente DEEPSEEK_API_KEY esta configurada?
2. Voce reiniciou o terminal/IDE apos configurar?

Como configurar no Windows (PowerShell):
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sua-chave-aqui", "User")

Depois reinicie o terminal!
""")
    else:
        if st.button("🚀 Gerar Insights com DeepSeek", use_container_width=True):
            with st.spinner("🤖 DeepSeek está analisando os dados..."):
                insights = gerar_insights_deepseek(df_filtrado, resultados)
                st.markdown("### 📋 Análise Gerada por IA")
                st.markdown(insights)
                
                st.download_button(
                    label="📥 Baixar Insights",
                    data=insights,
                    file_name="insights_churn_deepseek.txt",
                    mime="text/plain"
                )
        
        st.markdown("---")
        st.subheader("💬 Faça uma Pergunta Específica para DeepSeek")
        
        pergunta = st.text_area("O que você gostaria de saber sobre os dados?", placeholder="Ex: Por que clientes da Alemanha cancelam mais?", height=100)
        
        if pergunta and st.button("🔍 Perguntar à DeepSeek"):
            with st.spinner("🤖 DeepSeek está pensando..."):
                prompt_pergunta = f"""
Com base nos dados de churn abaixo, responda à pergunta do usuário.

DADOS:
- Taxa de churn: {resultados['taxa_churn']:.1f}%
- Idade média churn: {df_filtrado[df_filtrado['Exited']==1]['Age'].mean():.1f} anos
- Saldo médio churn: R$ {df_filtrado[df_filtrado['Exited']==1]['Balance'].mean():.2f}
- Churn por país: {df_filtrado.groupby('Geography')['Exited'].mean().to_dict()}
- % ativos (churn): {df_filtrado[df_filtrado['Exited']==1]['IsActiveMember'].mean()*100:.1f}%
- % ativos (não churn): {df_filtrado[df_filtrado['Exited']==0]['IsActiveMember'].mean()*100:.1f}%

PERGUNTA: {pergunta}

RESPONDA de forma clara, baseada nos dados, em português.
"""
                try:
                    response = openai.ChatCompletion.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "Você é um especialista em análise de dados e churn. Responda sempre em português."},
                            {"role": "user", "content": prompt_pergunta}
                        ],
                        temperature=0.5,
                        max_tokens=1000
                    )
                    st.success("✅ Resposta gerada por DeepSeek:")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

with tab3:
    st.header("📝 Relatório Completo de Análise")
    
    relatorio = f"""
# 📊 RELATÓRIO DE ANÁLISE DE CHURN

## 1. Resumo Executivo
- **Total de clientes analisados:** {len(df_filtrado):,}
- **Clientes que cancelaram:** {df_filtrado['Exited'].sum():,}
- **Taxa de churn:** {resultados['taxa_churn']:.1f}%

## 2. Perfil do Cliente que Cancela

| Característica | Quem Fica | Quem Cancela | Diferença |
| :--- | :--- | :--- | :--- |
| Idade | {df_filtrado[df_filtrado['Exited']==0]['Age'].mean():.1f} anos | {df_filtrado[df_filtrado['Exited']==1]['Age'].mean():.1f} anos | +{df_filtrado[df_filtrado['Exited']==1]['Age'].mean() - df_filtrado[df_filtrado['Exited']==0]['Age'].mean():.1f} anos |
| Saldo | R$ {df_filtrado[df_filtrado['Exited']==0]['Balance'].mean():.2f} | R$ {df_filtrado[df_filtrado['Exited']==1]['Balance'].mean():.2f} | +R$ {df_filtrado[df_filtrado['Exited']==1]['Balance'].mean() - df_filtrado[df_filtrado['Exited']==0]['Balance'].mean():.2f} |
| Atividade | {df_filtrado[df_filtrado['Exited']==0]['IsActiveMember'].mean()*100:.1f}% | {df_filtrado[df_filtrado['Exited']==1]['IsActiveMember'].mean()*100:.1f}% | -{df_filtrado[df_filtrado['Exited']==0]['IsActiveMember'].mean()*100 - df_filtrado[df_filtrado['Exited']==1]['IsActiveMember'].mean()*100:.1f} pp |
| Produtos | {df_filtrado[df_filtrado['Exited']==0]['NumOfProducts'].mean():.2f} | {df_filtrado[df_filtrado['Exited']==1]['NumOfProducts'].mean():.2f} | -{df_filtrado[df_filtrado['Exited']==0]['NumOfProducts'].mean() - df_filtrado[df_filtrado['Exited']==1]['NumOfProducts'].mean():.2f} |

## 3. Testes de Hipóteses (p-valor)

| Fator | p-valor | Significativo? |
| :--- | :--- | :--- |
| Idade | {resultados['p_idade']:.5f} | {'✅ Sim' if resultados['p_idade'] < 0.05 else '❌ Não'} |
| Saldo | {resultados['p_balance']:.5f} | {'✅ Sim' if resultados['p_balance'] < 0.05 else '❌ Não'} |
| Atividade | {resultados['p_active']:.5f} | {'✅ Sim' if resultados['p_active'] < 0.05 else '❌ Não'} |
| Score de Crédito | {resultados['p_credit']:.5f} | {'✅ Sim' if resultados['p_credit'] < 0.05 else '❌ Não'} |

## 4. Churn por País

| País | Taxa de Churn |
| :--- | :--- |
{chr(10).join([f"| {pais} | {taxa:.1f}%" for pais, taxa in df_filtrado.groupby('Geography')['Exited'].mean().items()])}

## 5. Features Mais Importantes

| Feature | Importância |
| :--- | :--- |
{chr(10).join([f"| {feature} | {importancia*100:.1f}%" for feature, importancia in resultados['feature_importance'].items()])}

## 6. Recomendações Estratégicas

### 👴 Clientes com 50+ anos
- **Problema:** Alta taxa de churn (56%)
- **Estratégia:** Programa de retenção específico com consultoria financeira personalizada
- **Ação:** Criar benefícios exclusivos para aposentadoria

### 📱 Clientes Inativos
- **Problema:** 2x mais chance de cancelar
- **Estratégia:** Campanha de reengajamento
- **Ação:** "Volte e ganhe benefícios" + ofertas especiais

### 🏦 Clientes com 1 produto
- **Problema:** Menos laços com o banco
- **Estratégia:** Cross-selling
- **Ação:** Ofertas de cartão de crédito, investimentos e seguros

### 🌍 Clientes na Alemanha
- **Problema:** Taxa de churn 2x maior (32%)
- **Estratégia:** Investigar causas locais
- **Ação:** Adaptar serviços ao mercado alemão

## 7. Próximos Passos Sugeridos

1. **Validar causas** - Investigar por que clientes mais velhos cancelam
2. **Testar ações** - Implementar campanhas piloto para inativos
3. **Monitorar resultados** - Acompanhar métricas semanalmente
4. **Refinar modelo** - Coletar mais dados para melhorar predições

---
*📅 Relatório gerado automaticamente em {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}*
*🔧 Ferramenta: Dashboard de Churn com DeepSeek AI*
*📊 Dados: Churn_Modelling.csv (Kaggle)*
"""
    
    st.markdown(relatorio)
    st.download_button(
        label="📥 Baixar Relatório Completo",
        data=relatorio,
        file_name=f"relatorio_churn_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("🚀 Dashboard desenvolvido com Streamlit + DeepSeek AI | Projeto de Análise de Churn")