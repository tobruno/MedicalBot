import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Wczytanie danych z podanego ciągu tekstowego


df = pd.read_csv("ragas/ragas_automated_summary_overall_liveqa.csv")

df = df.set_index('Metoda')

# 2. Transpozycja danych (.T), aby metryki były na osi X, a metody w legendzie (tak jak na zdjęciu)
df_transposed = df.T

# Tłumaczenie nazw metryk na czytelny format angielski z wielkich liter
df_transposed.index = df_transposed.index.map({
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "nv_context_relevance": "Context Relevance"
})

# 3. Generowanie wykresu dokładnie za pomocą kodu bazowego z Matplotlib (styl ze zdjęcia)
fig, ax = plt.subplots(figsize=(10, 6))

# Użycie domyślnej palety Matplotlib (klasyczny niebieski, pomarańczowy, zielony ze zdjęcia)
df_transposed.plot(kind='bar', ax=ax, width=0.6)

# Dodanie wartości nad słupkami (czcionka regularna, format %.2f)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

# Wszystkie opisy, tytuły i osie w języku angielskim
plt.title('Overall RAGAS Metrics Comparison', fontsize=14, pad=15)
plt.ylabel('Average Score', fontsize=12)
plt.xlabel('') # Ukrywamy podpis osi X, bo nazwy metryk są jasne
plt.ylim(0, 1.1)

# Obrócenie napisów na osi X o 15 stopni (dokładnie jak w kodzie i na zdjęciu)
plt.xticks(rotation=15, ha='right')

# Dodanie identycznej przerywanej siatki w tle dla osi Y
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Legenda po angielsku
plt.legend(title='Method', loc='upper left')

plt.tight_layout()
# 5. Zapis oraz wyświetlenie wykresu
output_path = "ragas/ragas_metrics_comparison_liveqa_en.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"🖼️ Chart successfully generated and saved to: {output_path}")

plt.show()