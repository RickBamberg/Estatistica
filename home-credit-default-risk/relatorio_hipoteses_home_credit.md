# Análise de Hipóteses — Home Credit Default Risk

## 1. Objetivo

Este projeto investiga quais fatores estão associados à inadimplência (`TARGET = 1`) de clientes no dataset **Home Credit Default Risk**, testando hipóteses organizadas em cinco eixos temáticos: demográfico, financeiro, emprego/estabilidade, histórico externo de crédito (bureau) e comportamento de pagamento interno.

## 2. Base de dados

- Fonte: dataset da competição Kaggle "Home Credit Default Risk"
- Tabela principal: `application_train.csv` (307.511 clientes, `TARGET` = 0 pagou / 1 = default)
- `application_test.csv` **não** foi usado nesta análise — não possui `TARGET`, serve apenas para submissão de modelo
- Taxa de inadimplência da base: **8,07%**

### 2.1. Relacionamento entre as tabelas

Todas as tabelas auxiliares se conectam à tabela principal por `SK_ID_CURR`, e as tabelas de histórico interno se conectam entre si por `SK_ID_PREV`:

- `application_train` (`SK_ID_CURR`) → `bureau` (`SK_ID_CURR`) → `bureau_balance` (`SK_ID_BUREAU`)
- `application_train` (`SK_ID_CURR`) → `previous_application` (`SK_ID_CURR`, `SK_ID_PREV`) → `POS_CASH_balance`, `credit_card_balance`, `installments_payments` (`SK_ID_PREV`)

## 3. Tratamento de outliers e valores sentinela

Antes dos testes, a base passou pelos seguintes ajustes:

| Variável | Problema identificado | Tratamento |
|---|---|---|
| `DAYS_EMPLOYED` | Valor sentinela `365243` (~1000 anos) para aposentados/desempregados — código de erro do sistema, não outlier real | Convertido para `NaN` |
| `DAYS_BIRTH`, `DAYS_EMPLOYED`, etc. | Valores negativos (dias antes da aplicação) | Convertidos para valor absoluto; criadas `AGE_YEARS` e `EMPLOYED_YEARS` |
| `AMT_INCOME_TOTAL` | Outliers extremos (rendas declaradas de dezenas de milhões) | Winsorização no percentil 99,5 (1.446 valores limitados) |
| `CNT_CHILDREN`, `CNT_FAM_MEMBERS` | Valores absurdos (ex: 19 filhos) | Winsorização no percentil 99,9 |

## 4. Metodologia dos testes

- **Variáveis categóricas vs. `TARGET`**: teste **qui-quadrado** de independência, com **Cramér's V** como tamanho de efeito (0 = nenhuma associação, 1 = associação perfeita)
- **Variáveis numéricas vs. `TARGET`**: teste **Mann-Whitney U** (não assume normalidade — variáveis financeiras costumam ser assimétricas), com **r rank-biserial** como tamanho de efeito (-1 a 1)

Com ~300 mil observações, praticamente qualquer diferença gera p-valor muito baixo. Por isso, o **tamanho de efeito** foi o critério decisivo para julgar relevância prática, não apenas o p-valor:

| Cramér's V | Interpretação | \| r \| | Interpretação |
|---|---|---|---|
| < 0,10 | desprezível | < 0,10 | desprezível |
| 0,10–0,20 | fraco | 0,10–0,30 | pequeno |
| 0,20–0,40 | moderado | 0,30–0,50 | moderado |
| > 0,40 | forte | > 0,50 | grande |

## 5. Resultados por eixo

### 5.1. Demográficas

| Hipótese | p-valor | Efeito | Classificação |
|---|---|---|---|
| Idade influencia default | 0 | -0,166 | pequeno |
| Escolaridade influencia default | 2,45e-219 | 0,058 | desprezível |
| Número de filhos influencia default | 8,77e-29 | 0,034 | desprezível |
| Estado civil influencia default | 7,75e-107 | 0,041 | desprezível |

Clientes inadimplentes são mais jovens (mediana 39,1 vs. 43,5 anos).

### 5.2. Financeiras

| Hipótese | p-valor | Efeito | Classificação |
|---|---|---|---|
| Razão crédito/renda influencia default | 0,275 | -0,004 | **não significativo** |
| Tipo de contrato influencia default | 1,02e-65 | 0,031 | desprezível |
| Tipo de renda influencia default | 1,93e-266 | 0,064 | desprezível |
| EXT_SOURCE_1 influencia default | 0 | -0,331 | moderado |
| EXT_SOURCE_2 influencia default | 0 | -0,312 | moderado |
| EXT_SOURCE_3 influencia default | 0 | -0,359 | moderado |

Os três scores externos (`EXT_SOURCE`) são as variáveis com maior poder discriminativo de toda a análise.

### 5.3. Emprego e estabilidade

| Hipótese | p-valor | Efeito | Classificação |
|---|---|---|---|
| Tempo de emprego influencia default | 0 | -0,165 | pequeno |
| Ocupação influencia default | 3,78e-288 | 0,082 | desprezível |

### 5.4. Histórico externo (bureau)

| Hipótese | p-valor | Efeito | Classificação |
|---|---|---|---|
| Ter atraso no bureau influencia default | 2,44e-63 | 0,030 | desprezível |
| Quantidade de créditos ativos influencia default | 8,55e-56 | 0,059 | desprezível |

### 5.5. Comportamento de pagamento interno

| Hipótese | p-valor | Efeito | Classificação |
|---|---|---|---|
| Ter pedido anterior recusado influencia default | 1,83e-222 | 0,057 | desprezível |
| Atraso médio em parcelas influencia default | 3,64e-106 | 0,085 | desprezível |

## 6. Conclusões

1. **Nenhuma variável isolada tem efeito forte.** O máximo observado foi "moderado", nos três `EXT_SOURCE`. Isso é esperado em dados reais de crédito: se uma única variável explicasse bem o default, o problema de scoring seria trivial.
2. **Significância estatística ≠ relevância prática.** Quase todas as hipóteses testadas foram "estatisticamente significativas" (p-valor baixíssimo), simplesmente por causa do tamanho da amostra. O tamanho de efeito foi o que realmente separou sinal de ruído — a maioria das variáveis demográficas e categóricas, apesar de "significativas", teve efeito desprezível.
3. **A hipótese da razão crédito/renda não se sustentou** — único caso sem significância estatística e com efeito nulo. Contraintuitivo frente ao senso comum de crédito, mas consistente com o fato de que renda declarada é ruidosa e o Home Credit provavelmente já filtra casos extremos na aprovação inicial.
4. **Risco de crédito é multifatorial.** Um modelo eficaz precisa necessariamente combinar múltiplas variáveis — nenhuma regra simples baseada em 1–2 variáveis teria bom poder preditivo.

## 7. Insights gerados por LLM (DeepSeek)

*Gerado automaticamente a partir dos resultados consolidados dos testes, usando a API da DeepSeek.*

### Resumo executivo

A taxa de inadimplência na base analisada é de 8,07% (aproximadamente 1 em cada 12 clientes). O perfil típico do cliente inadimplente é mais jovem (39 anos vs. 43,5 anos), com menos tempo de vínculo empregatício (3,4 anos vs. 4,6 anos) e com pontuações de fontes externas de crédito (EXT_SOURCE) significativamente mais baixas — indicando menor histórico de solvência comprovada.

As variáveis com maior poder discriminativo são justamente as pontuações externas (EXT_SOURCE_1, _2 e _3), com efeitos moderados e negativos: quanto menor a pontuação, maior o risco. Embora diversas variáveis sociodemográficas e comportamentais sejam estatisticamente significativas, a maioria apresenta impacto prático pequeno, sendo importante priorizar os fatores com efeito real sobre o risco.

### Principais insights

- **Pontuações externas são o sinal mais forte de risco**: EXT_SOURCE_3 tem efeito de -0,36 — o maior entre todas as variáveis.
- **Cliente inadimplente é 4,4 anos mais jovem** (39,1 vs. 43,5 anos), efeito de -0,17.
- **Estabilidade empregatícia importa**: inadimplentes possuem 1,2 anos a menos de tempo de emprego, efeito de -0,16.
- **Tipo de renda e ocupação são relevantes**, mas com efeitos pequenos (0,06 e 0,08).
- **Atraso médio em parcelas é o comportamento mais preditivo entre os históricos** (efeito de 0,09), superando número de créditos ativos e pedidos recusados.

### Recomendações de ação

| Perfil de risco | Estratégia recomendada |
|---|---|
| Clientes jovens (< 30 anos) com pouco tempo de emprego (< 2 anos) | Exigir garantias adicionais ou avalistas; oferecer produtos menores com aumento progressivo de crédito |
| Clientes com EXT_SOURCE baixo (< 0,40) | Scorecard mais rigoroso; considerar dados alternativos complementares |
| Ocupações informais ou renda instável | Comprovação de renda estendida; parcelas menores com prazos mais curtos |
| Histórico de atraso médio em parcelas elevado | Condicionar novo crédito a período de bom comportamento; alertas preventivos de cobrança |
| Múltiplos créditos ativos ou pedidos recusados recentes | Limite de comprometimento de renda; cruzamento com bureau para sinalizar sobre-alavancagem |

### Insights não óbvios

- **Significância estatística ≠ relevância prática**: variáveis como estado civil e número de filhos são "significativas" apenas pelo tamanho da amostra — seus efeitos práticos são desprezíveis.
- **Razão crédito/renda não é significativa**, sugerindo que a estabilidade da renda (tipo de renda, comportamento de pagamento) importa mais que o valor comprometido em si.
- **O comportamento do cliente parece pesar mais que o produto contratado** — tipo de contrato tem efeito fraco frente às variáveis comportamentais.
- **Pedido recusado é sinalizador, não causa** — provavelmente reflete um risco que já existia antes da recusa, já capturado por outras variáveis do bureau.

---

*Análise realizada com Python (pandas, scipy), testes de hipótese com correção via tamanho de efeito, e geração de insights complementares via LLM (DeepSeek).*
