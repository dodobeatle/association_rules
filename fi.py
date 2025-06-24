import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Cargar el dataset
df = pd.read_excel('online_retail_2.xlsx')

# Preprocesamiento de datos
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

# Crear matriz de transacciones (one-hot encoding)
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Convertir cantidades a booleanos (1: comprado, 0: no comprado)
basket_sets = basket.map(lambda x: True if x > 0 else False)

# Generar conjuntos frecuentes con soporte mínimo de 0.01
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Ordenar reglas por lift
rules = rules.sort_values('lift', ascending=False)

# Mostrar los primeros resultados
print("Conjuntos frecuentes más comunes:")
print(frequent_itemsets.head())
print("\nReglas de asociación más fuertes:")
print(rules.head())
