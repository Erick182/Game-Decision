from nba_api.stats.endpoints import leaguedashplayerstats 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
import seaborn as sns 
import matplotlib.pyplot as plt 

 
 # Coletar estatísticas de jogadores da temporada regular 

player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2023-24', season_type_all_star='Regular Season') 

df = player_stats.get_data_frames()[0]  


df.to_csv("dados_jogadores_nba.csv", index=False) 

print (df) 

# Colunas que você quer manter 

colunas_importantes = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'FG_PCT', 'FT_PCT', 'AST', 'TOV'] 

 

# Verificar quais dessas colunas existem no DataFrame 

colunas_existentes = [col for col in colunas_importantes if col in df.columns] 

 

print(f"Colunas solicitadas: {colunas_importantes}") 

print(f"Colunas disponíveis: {df.columns.tolist()}") 

print(f"Colunas que serão usadas: {colunas_existentes}") 


# Carregar os dados 

df = pd.read_csv("dados_jogadores_nba.csv") 

 

# Remover colunas que não serão usadas de imediato 

colunas_importantes = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'FG_PCT', 'FT_PCT', 'AST', 'TOV'] 

df = df[colunas_importantes] 

 

# Tratar valores ausentes 

df.dropna(inplace=True) 

# Salvar dados limpos 

df.to_csv("dados_limpos_nba.csv", index=False) 

print(df) 

df = pd.read_csv("dados_limpos_nba.csv") 

 

# Correlação 

corr = df[['FG_PCT', 'FT_PCT', 'AST', 'TOV']].corr() 

sns.heatmap(corr, annot=True) 

plt.title("Correlação entre variáveis") 

plt.show() 

 

# Gráfico de dispersão Assistências vs Turnovers 

sns.scatterplot(data=df, x='AST', y='TOV') 

plt.title("Assistências vs Turnovers") 

plt.show() 

# Carregar dados 

df = pd.read_csv("dados_limpos_nba.csv") 

 

# Criar coluna-alvo: jogada bem-sucedida (aproveitamento FG% > 0.45) 

df['LABEL'] = df['FG_PCT'].apply(lambda x: 1 if x > 0.45 else 0) 

 

# Selecionar variáveis 

X = df[['FT_PCT', 'AST', 'TOV', 'MIN']] 

y = df['LABEL'] 

 

# Treinar modelo 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

modelo = LogisticRegression() 

modelo.fit(X_train, y_train) 

 

# Avaliar 

y_pred = modelo.predict(X_test) 

acc = accuracy_score(y_test, y_pred) 

print(f"Acurácia do modelo: {acc:.2%}") 

 
# Reutilizar os dados 

X = df[['FT_PCT', 'AST', 'TOV', 'MIN']] 

y = df['LABEL'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

 

# Modelos 

modelos = { 

    "Logistic Regression": LogisticRegression(), 

    "Random Forest": RandomForestClassifier(random_state=42), 

    "XGBoost": XGBClassifier(eval_metric='logloss') 

} 

 

# Resultados 

resultados = [] 

 

for nome, modelo in modelos.items(): 

    modelo.fit(X_train, y_train) 

    y_pred = modelo.predict(X_test) 

    acc = accuracy_score(y_test, y_pred) 

    prec = precision_score(y_test, y_pred) 

    rec = recall_score(y_test, y_pred) 

    resultados.append((nome, acc, prec, rec)) 

 

# Tabela final 

df_resultados = pd.DataFrame(resultados, columns=["Modelo", "Acurácia", "Precisão", "Recall"]) 

print(df_resultados) 


# Normalizar dados 

features_cluster = ['FT_PCT', 'AST', 'TOV', 'MIN'] 

X_cluster = df[features_cluster] 

scaler = StandardScaler() 

X_scaled = scaler.fit_transform(X_cluster) 

 

# KMeans 

kmeans = KMeans(n_clusters=3, random_state=42) 

df['Cluster'] = kmeans.fit_predict(X_scaled) 

 

# Visualização dos clusters 

plt.figure(figsize=(8, 5)) 

sns.scatterplot(data=df, x='MIN', y='FT_PCT', hue='Cluster', palette='Set2') 

plt.title("Clusters de Jogadores por MIN e FT%") 

plt.xlabel("Minutos em quadra") 

plt.ylabel("Free Throw %") 

plt.show() 

 
# Simulação de gráfico: desempenho médio por tempo em quadra 

plt.figure(figsize=(10, 6)) 

sns.lineplot(data=df, x='MIN', y='FG_PCT') 

plt.title("Desempenho Médio (FG%) por Tempo em Quadra") 

plt.xlabel("Minutos em quadra") 

plt.ylabel("Field Goal %") 

plt.show() 

 