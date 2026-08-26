"""
Análise de Hipóteses - Home Credit Default Risk
=================================================
1. Carrega e trata a base (outliers/valores sentinela)
2. Testa hipóteses por eixo temático (demográfico, financeiro, emprego, bureau, pagamento interno)

Ajuste os caminhos (CAMINHO_TRAIN, CAMINHO_BUREAU, etc.) conforme sua pasta local.
"""

import os
import pandas as pd
import numpy as np
import requests
from scipy import stats

pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# 1. CARREGAMENTO
# ---------------------------------------------------------------------------
CAMINHO_TRAIN = "data/application_train.csv"
CAMINHO_BUREAU = "data/bureau.csv"
CAMINHO_INSTALLMENTS = "data/installments_payments.csv"
CAMINHO_PREVIOUS = "data/previous_application.csv"

df = pd.read_csv(CAMINHO_TRAIN)
print(f"Shape original: {df.shape}")


# ---------------------------------------------------------------------------
# 2. TRATAMENTO DE OUTLIERS / VALORES SENTINELA
# ---------------------------------------------------------------------------
def tratar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # DAYS_EMPLOYED tem o valor sentinela 365243 (~1000 anos) para
    # aposentados/desempregados -- não é outlier real, é código de erro do
    # sistema de origem. Convertido para NaN.
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH vêm
    # negativos (dias antes da aplicação). Convertendo para anos/positivo
    # facilita leitura e evita erro de sinal em comparações.
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]:
        if col in df.columns:
            df[col] = df[col].abs()
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = df["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"] / 365.25

    # AMT_INCOME_TOTAL tem outliers extremos (ex: valores acima de 100 milhões
    # em poucas linhas) que distorcem qualquer teste. Cap no percentil 99.5.
    if "AMT_INCOME_TOTAL" in df.columns:
        limite = df["AMT_INCOME_TOTAL"].quantile(0.995)
        antes = (df["AMT_INCOME_TOTAL"] > limite).sum()
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=limite)
        print(f"AMT_INCOME_TOTAL: {antes} valores acima do p99.5 ({limite:,.0f}) limitados (winsorização)")

    # CNT_CHILDREN e CNT_FAM_MEMBERS têm alguns valores absurdos (ex: 19 filhos)
    for col in ["CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
        if col in df.columns:
            limite = df[col].quantile(0.999)
            antes = (df[col] > limite).sum()
            df[col] = df[col].clip(upper=limite)
            if antes:
                print(f"{col}: {antes} valores acima do p99.9 limitados")

    # Razão crédito/renda -- feature derivada útil para a hipótese financeira
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    return df


df = tratar_outliers(df)
print(f"Shape após tratamento: {df.shape}\n")


# ---------------------------------------------------------------------------
# 3. CONFIGURAÇÃO DO DEEPSEEK
# ---------------------------------------------------------------------------
class DeepSeekClient:
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com/chat/completions"

    def chat(self, messages, temperature=0.7, max_tokens=2000):
        """Envia uma mensagem para a DeepSeek"""
        if not self.api_key:
            return "❌ Chave DEEPSEEK_API_KEY não encontrada!"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            resultado = response.json()
            return resultado['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ Erro na API: {str(e)}"


# Inicializar cliente
cliente = DeepSeekClient()


# ---------------------------------------------------------------------------
# 4. FUNÇÕES DE TESTE
# ---------------------------------------------------------------------------
def interpretar_cramers_v(v):
    if v < 0.10:
        return "efeito desprezível"
    elif v < 0.20:
        return "efeito fraco"
    elif v < 0.40:
        return "efeito moderado"
    else:
        return "efeito forte"


def testar_categorica(df, coluna, target="TARGET", alpha=0.05):
    """Qui-quadrado: associação entre variável categórica e TARGET.
    Tamanho de efeito: Cramér's V (0=nenhuma associação, 1=associação perfeita).
    Necessário porque com ~300k linhas quase tudo dá p-valor baixo -- o V mostra
    se a associação é forte o suficiente para importar na prática."""
    tabela = pd.crosstab(df[coluna], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(tabela)
    n = tabela.sum().sum()
    k = min(tabela.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else np.nan
    print(f"[Qui-quadrado] {coluna} vs {target}")
    print(f"  chi2={chi2:.2f}  p-valor={p:.4g}  -> {'REJEITA H0 (associação significativa)' if p < alpha else 'não rejeita H0'}")
    print(f"  Cramér's V={cramers_v:.4f}  -> {interpretar_cramers_v(cramers_v)}\n")
    return p, cramers_v


def interpretar_r(r):
    r = abs(r)
    if r < 0.10:
        return "efeito desprezível"
    elif r < 0.30:
        return "efeito pequeno"
    elif r < 0.50:
        return "efeito moderado"
    else:
        return "efeito grande"


def testar_numerica(df, coluna, target="TARGET", alpha=0.05):
    """Mann-Whitney U: compara distribuição da variável numérica entre TARGET=0 e TARGET=1.
    Preferido ao teste t por não assumir normalidade (essas variáveis costumam ser assimétricas).
    Tamanho de efeito: r rank-biserial (-1 a 1). Com ~300k linhas o p-valor quase
    sempre vem ~0 -- o r mostra se a diferença é grande o suficiente pra importar."""
    grupo0 = df.loc[df[target] == 0, coluna].dropna()
    grupo1 = df.loc[df[target] == 1, coluna].dropna()
    n0, n1 = len(grupo0), len(grupo1)
    stat, p = stats.mannwhitneyu(grupo0, grupo1, alternative="two-sided")
    # r rank-biserial = 1 - (2*U) / (n0*n1), onde U é a estatística do grupo0
    r = 1 - (2 * stat) / (n0 * n1)
    print(f"[Mann-Whitney] {coluna} vs {target}")
    print(f"  mediana TARGET=0: {grupo0.median():.2f}  |  mediana TARGET=1: {grupo1.median():.2f}")
    print(f"  p-valor={p:.4g}  -> {'REJEITA H0 (diferença significativa)' if p < alpha else 'não rejeita H0'}")
    print(f"  r rank-biserial={r:.4f}  -> {interpretar_r(r)}\n")
    return p, r


# ---------------------------------------------------------------------------
# 5. TESTE DAS HIPÓTESES POR EIXO
# ---------------------------------------------------------------------------
resultados_testes = {}

print("=" * 70)
print("EIXO 1: DEMOGRÁFICAS")
print("=" * 70)
p, r = testar_numerica(df, "AGE_YEARS")
resultados_testes["Idade"] = {"p": p, "efeito": r, "tipo": "numerica"}

p, v = testar_categorica(df, "NAME_EDUCATION_TYPE")
resultados_testes["Escolaridade"] = {"p": p, "efeito": v, "tipo": "categorica"}

p, r = testar_numerica(df, "CNT_CHILDREN")
resultados_testes["Número de filhos"] = {"p": p, "efeito": r, "tipo": "numerica"}

p, v = testar_categorica(df, "NAME_FAMILY_STATUS")
resultados_testes["Estado civil"] = {"p": p, "efeito": v, "tipo": "categorica"}

print("=" * 70)
print("EIXO 2: FINANCEIRAS")
print("=" * 70)
p, r = testar_numerica(df, "CREDIT_INCOME_RATIO")
resultados_testes["Razão crédito/renda"] = {"p": p, "efeito": r, "tipo": "numerica"}

p, v = testar_categorica(df, "NAME_CONTRACT_TYPE")
resultados_testes["Tipo de contrato"] = {"p": p, "efeito": v, "tipo": "categorica"}

p, v = testar_categorica(df, "NAME_INCOME_TYPE")
resultados_testes["Tipo de renda"] = {"p": p, "efeito": v, "tipo": "categorica"}

for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    if col in df.columns:
        p, r = testar_numerica(df, col)
        resultados_testes[col] = {"p": p, "efeito": r, "tipo": "numerica"}

print("=" * 70)
print("EIXO 3: EMPREGO E ESTABILIDADE")
print("=" * 70)
p, r = testar_numerica(df, "EMPLOYED_YEARS")
resultados_testes["Tempo de emprego"] = {"p": p, "efeito": r, "tipo": "numerica"}

p, v = testar_categorica(df, "OCCUPATION_TYPE")
resultados_testes["Ocupação"] = {"p": p, "efeito": v, "tipo": "categorica"}

print("=" * 70)
print("EIXO 4: HISTÓRICO EXTERNO (BUREAU)")
print("=" * 70)

bureau = pd.read_csv(CAMINHO_BUREAU)

# bureau tem N linhas por cliente (um por crédito externo) -- precisa agregar
# para 1 linha por SK_ID_CURR antes de juntar com df.
agg_bureau = bureau.groupby("SK_ID_CURR").agg(
    MAX_CREDIT_DAY_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
    QTD_CREDITOS_BUREAU=("SK_ID_BUREAU", "count"),
    QTD_CREDITOS_ATIVOS=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
).reset_index()

df = df.merge(agg_bureau, on="SK_ID_CURR", how="left")
# Cliente sem registro no bureau = sem histórico externo, não é atraso/crédito
df["MAX_CREDIT_DAY_OVERDUE"] = df["MAX_CREDIT_DAY_OVERDUE"].fillna(0)
df["QTD_CREDITOS_BUREAU"] = df["QTD_CREDITOS_BUREAU"].fillna(0)
df["QTD_CREDITOS_ATIVOS"] = df["QTD_CREDITOS_ATIVOS"].fillna(0)

# Hipótese: cliente com histórico de atraso externo (mesmo que só uma vez) tem mais default
df["TEVE_ATRASO_BUREAU"] = (df["MAX_CREDIT_DAY_OVERDUE"] > 0).map({True: "Sim", False: "Não"})

p, v = testar_categorica(df, "TEVE_ATRASO_BUREAU")
resultados_testes["Teve atraso no bureau"] = {"p": p, "efeito": v, "tipo": "categorica"}

p, r = testar_numerica(df, "QTD_CREDITOS_ATIVOS")
resultados_testes["Qtd. créditos ativos"] = {"p": p, "efeito": r, "tipo": "numerica"}

print("=" * 70)
print("EIXO 5: COMPORTAMENTO DE PAGAMENTO INTERNO")
print("=" * 70)

previous = pd.read_csv(CAMINHO_PREVIOUS)
installments = pd.read_csv(CAMINHO_INSTALLMENTS)

# previous_application: teve pedido anterior recusado?
agg_previous = previous.groupby("SK_ID_CURR").agg(
    QTD_PEDIDOS_RECUSADOS=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
).reset_index()
df = df.merge(agg_previous, on="SK_ID_CURR", how="left")
df["QTD_PEDIDOS_RECUSADOS"] = df["QTD_PEDIDOS_RECUSADOS"].fillna(0)
df["TEVE_PEDIDO_RECUSADO"] = (df["QTD_PEDIDOS_RECUSADOS"] > 0).map({True: "Sim", False: "Não"})

# installments_payments: atraso médio entre data esperada e data real de pagamento
# Nota: installments_payments.csv já vem com SK_ID_CURR próprio (não precisa
# recuperar via merge com previous_application -- fazer esse merge cria
# SK_ID_CURR_x/SK_ID_CURR_y por causa da coluna duplicada, daí o KeyError)
installments["ATRASO_DIAS"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]

agg_installments = installments.groupby("SK_ID_CURR").agg(
    ATRASO_MEDIO_DIAS=("ATRASO_DIAS", "mean"),
).reset_index()
df = df.merge(agg_installments, on="SK_ID_CURR", how="left")
# Cliente sem histórico de parcelas anteriores fica com NaN (tratado como ausente, não zero)

p, v = testar_categorica(df, "TEVE_PEDIDO_RECUSADO")
resultados_testes["Teve pedido recusado"] = {"p": p, "efeito": v, "tipo": "categorica"}

p, r = testar_numerica(df, "ATRASO_MEDIO_DIAS")
resultados_testes["Atraso médio em parcelas"] = {"p": p, "efeito": r, "tipo": "numerica"}


# ---------------------------------------------------------------------------
# 6. FUNÇÃO PARA GERAR INSIGHTS
# ---------------------------------------------------------------------------
def gerar_insights_deepseek(df, resultados_testes):
    """Gera insights usando a API da DeepSeek a partir dos resultados dos testes de hipótese.

    resultados_testes: dict no formato
        {"NOME_VARIAVEL": {"p": <p-valor>, "efeito": <Cramér's V ou r>, "tipo": "categorica"|"numerica"}}
    """

    n_total = len(df)
    n_default = df["TARGET"].sum()
    taxa_default = n_default / n_total * 100

    # Monta o bloco de resultados dos testes dinamicamente a partir do que já foi calculado
    linhas_testes = []
    for var, r in resultados_testes.items():
        status = "significativo" if r["p"] < 0.05 else "não significativo"
        linhas_testes.append(
            f"- {var}: p-valor={r['p']:.4g} ({status}) | tamanho de efeito={r['efeito']:.4f}"
        )
    bloco_testes = "\n".join(linhas_testes)

    prompt = f"""
Você é um especialista em risco de crédito (Credit Risk Analytics).
Analise os resultados de um estudo de hipóteses sobre inadimplência de clientes
(dataset Home Credit Default Risk) e gere insights acionáveis.

DADOS GERAIS:
- Total de clientes analisados: {n_total}
- Clientes com default (TARGET=1): {n_default}
- Taxa de default: {taxa_default:.2f}%

PERFIL DO CLIENTE QUE DÁ DEFAULT (mediana):
- Idade: {df.loc[df['TARGET']==1, 'AGE_YEARS'].median():.1f} anos (vs {df.loc[df['TARGET']==0, 'AGE_YEARS'].median():.1f} anos de quem paga)
- Tempo de emprego: {df.loc[df['TARGET']==1, 'EMPLOYED_YEARS'].median():.1f} anos (vs {df.loc[df['TARGET']==0, 'EMPLOYED_YEARS'].median():.1f} anos)
- EXT_SOURCE_3: {df.loc[df['TARGET']==1, 'EXT_SOURCE_3'].median():.2f} (vs {df.loc[df['TARGET']==0, 'EXT_SOURCE_3'].median():.2f})

TESTES DE HIPÓTESE (p-valor e tamanho de efeito):
{bloco_testes}

Com base nesses dados, forneça:
1. Resumo Executivo (2-3 parágrafos em linguagem simples)
2. Principais Insights (4-5 bullet points com números concretos)
3. Recomendações de Ação (estratégias práticas de mitigação de risco para cada perfil)
4. Insights Não Óbvios (padrões interessantes que não estão na superfície, incluindo
   o que significa uma variável ser "estatisticamente significativa" mas ter efeito
   prático desprezível nesse contexto)

Seja objetivo, prático, use números e evite jargões técnicos desnecessários.
Responda em português.
"""

    messages = [
        {"role": "system", "content": "Você é um especialista em risco de crédito e análise de inadimplência. Responda sempre em português."},
        {"role": "user", "content": prompt}
    ]

    return cliente.chat(messages)


# resultados_testes já foi montado automaticamente durante os testes dos 5 eixos acima
print("=" * 70)
print("GERANDO INSIGHTS COM DEEPSEEK...")
print("=" * 70)
insights = gerar_insights_deepseek(df, resultados_testes)
print(insights)
