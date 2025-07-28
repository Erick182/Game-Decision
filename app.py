# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Configuração da página
st.set_page_config(page_title="Análise Preditiva no Basquete", layout="wide")
st.title("🏀 Dashboard Analítico - Diagnóstico Tático no Basquete")

# Carregamento dos dados
df = pd.read_csv("dados_limpos_nba.csv")

# Exibição inicial dos dados
st.subheader("📊 Dados Brutos")
st.dataframe(df.head())

# Visualização 1: Correlação entre variáveis
st.subheader("📈 Correlação entre variáveis")
corr = df[['FG_PCT', 'FT_PCT', 'AST', 'TOV']].corr()
fig_corr, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
st.pyplot(fig_corr)

# Visualização 2: Assistências vs Turnovers
st.subheader("🎯 Assistências vs Turnovers")
fig_scatter = px.scatter(df, x="AST", y="TOV", color="FG_PCT",
                         title="Assistências x Erros com base no aproveitamento de arremessos")
st.plotly_chart(fig_scatter)

# Visualização 3: Diagnóstico Preditivo
st.subheader("🔍 Diagnóstico de Jogadas Bem-sucedidas (Modelo Preditivo)")

# Criar coluna-alvo
df['LABEL'] = df['FG_PCT'].apply(lambda x: 1 if x > 0.45 else 0)
X = df[['FT_PCT', 'AST', 'TOV', 'MIN']]
y = df['LABEL']

# Treinamento (simples para demo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Previsão e exibição
df['PREV'] = modelo.predict(X)
df['ALERTA'] = df['PREV'].apply(lambda x: "🔴 Ineficiente" if x == 0 else "🟢 Eficiente")
st.write("Legenda de alerta: Jogadas com FG% <= 45% são consideradas ineficientes.")

st.dataframe(df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN', 'FG_PCT', 'AST', 'TOV', 'PREV', 'ALERTA']].head(20))

# Visualização 4: Gráfico de desempenho médio por tempo
st.subheader("⏱️ Desempenho Médio (FG%) por Tempo em Quadra")
fig_fg = px.line(df.sort_values('MIN'), x='MIN', y='FG_PCT', title="FG% por Minutos em Quadra")
st.plotly_chart(fig_fg)

st.header("🎯 Filtro de Eficiência dos Jogadores")

filtro = st.selectbox(
    "Selecione o tipo de jogador para visualização:",
    ["Todos", "Apenas eficientes (🟢)", "Apenas ineficientes (🔴)"]
)

if filtro == "Apenas eficientes (🟢)":
    df_filtrado = df[df['PREV'] == 1]
elif filtro == "Apenas ineficientes (🔴)":
    df_filtrado = df[df['PREV'] == 0]
else:
    df_filtrado = df

# Tabela com os jogadores filtrados
st.subheader("📋 Jogadores selecionados")
st.dataframe(
    df_filtrado[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN', 'FG_PCT', 'AST', 'TOV', 'ALERTA']]
    .sort_values(by='FG_PCT', ascending=False)
    .reset_index(drop=True)
)

# -------------------------------
# 📊 Módulo: Gráfico por eficiência
# -------------------------------

st.header("📊 Gráfico de FG% dos Jogadores Filtrados")

top = st.slider("Quantos jogadores deseja exibir no gráfico?", 5, 30, 10)

df_top = df_filtrado[['PLAYER_NAME', 'FG_PCT']].sort_values(by='FG_PCT', ascending=False).head(top)

fig_bar = px.bar(
    df_top, x='PLAYER_NAME', y='FG_PCT', color='FG_PCT',
    title='FG% dos jogadores selecionados',
    labels={'PLAYER_NAME': 'Jogador', 'FG_PCT': 'Field Goal %'}
)

st.plotly_chart(fig_bar)

st.header("📉 Dispersão: Assistências vs Turnovers")

st.markdown(
    """
    Este gráfico mostra a relação entre **assistências (AST)** e **turnovers (TOV)** para cada jogador,
    com a **cor indicando o FG%**. Isso ajuda a identificar jogadores que contribuem com o time sem desperdiçar posses.
    """
)

fig_disp = px.scatter(
    df,
    x='AST',
    y='TOV',
    color='FG_PCT',
    hover_name='PLAYER_NAME',
    title='Assistências vs Turnovers com FG%',
    labels={
        'AST': 'Assistências',
        'TOV': 'Turnovers',
        'FG_PCT': 'Field Goal %'
    },
    color_continuous_scale='RdYlGn',
    height=600
)

st.plotly_chart(fig_disp)

st.header("⏱️ Desempenho por Tempo em Quadra (FG% x Minutos)")

st.markdown(
    """
    Este gráfico mostra como o aproveitamento de arremessos (**FG%**) varia de acordo com o tempo em quadra (**MIN**).
    Isso ajuda a identificar padrões de desgaste, eficiência em minutos reduzidos e perfil tático dos jogadores.
    """
)

# Arredondar os minutos para facilitar agrupamento (ex: grupos de 5 em 5 minutos)
df['MIN_ROUND'] = df['MIN'].apply(lambda x: round(x / 5) * 5)

# Agrupar e calcular média do FG% por faixa de minutos
df_fgmin = df.groupby('MIN_ROUND')['FG_PCT'].mean().reset_index()

# Gráfico de linha
fig_line = px.line(
    df_fgmin,
    x='MIN_ROUND',
    y='FG_PCT',
    markers=True,
    title='Média de FG% por Faixa de Minutos em Quadra',
    labels={'MIN_ROUND': 'Minutos (faixa)', 'FG_PCT': 'Aproveitamento FG%'}
)

st.plotly_chart(fig_line)



st.header("🔍 Diagnóstico Individual de Jogadores")

def diagnostico_jogador(df, nome_jogador):
    jogador = df[df['PLAYER_NAME'].str.lower() == nome_jogador.lower()]

    if jogador.empty:
        st.warning(f"Jogador '{nome_jogador}' não encontrado.")
        return

    # Usar médias por jogo (se tiver GP = Games Played)
    gp = jogador['GP'].mean() if 'GP' in df.columns else None

    if gp is None or gp == 0:
        st.warning("Número de jogos (GP) não disponível. Não é possível calcular médias por jogo corretamente.")
        return

    # Conversão para médias por jogo
    fg = jogador['FG_PCT'].mean()
    ft = jogador['FT_PCT'].mean() if 'FT_PCT' in df.columns else None
    ast_pg = jogador['AST'].sum() / gp
    tov_pg = jogador['TOV'].sum() / gp
    min_pg = jogador['MIN'].sum() / gp

    # Benchmarks realistas por jogo
    media_fg = 0.45
    media_ft = 0.75
    media_ast = 3.0
    media_tov = 2.0

    st.markdown(f"### Diagnóstico para **{nome_jogador.title()}**")

    if fg < media_fg:
        st.write(f"- ⚠️ FG% baixo: **{fg:.2f}** (ideal: > {media_fg})")
    else:
        st.write(f"- ✅ Bom FG%: **{fg:.2f}**")

    if ft is not None:
        if ft < media_ft:
            st.write(f"- ⚠️ FT% abaixo do ideal: **{ft:.2f}** (ideal: > {media_ft})")
        else:
            st.write(f"- ✅ Bom FT%: **{ft:.2f}**")

    if ast_pg < media_ast:
        st.write(f"- ⚠️ Poucas assistências por jogo: **{ast_pg:.1f}** (ideal: > {media_ast})")
    else:
        st.write(f"- ✅ Assistências por jogo: **{ast_pg:.1f}**")

    if tov_pg > media_tov:
        st.write(f"- ⚠️ Muitos turnovers por jogo: **{tov_pg:.1f}** (ideal: < {media_tov})")
    else:
        st.write(f"- ✅ Turnovers sob controle: **{tov_pg:.1f}**")

    st.write(f"- 🕒 Minutos por jogo: **{min_pg:.1f}**")

    st.markdown("---")
