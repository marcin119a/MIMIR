import json
from collections import Counter
import matplotlib.pyplot as plt
# 1. Wczytanie mapy sample -> primary_site
with open("data/primary_sites.json", "r") as f:
    primary_sites = json.load(f)  # dict: {sample_id: "Primary site"}
# 2. Policz częstości poszczególnych primary sites
site_counts = Counter(primary_sites.values())
# 3. Posortuj wg liczebności (opcjonalne, ale zwykle czytelniejsze)
sites, counts = zip(*sorted(site_counts.items(), key=lambda x: x[1], reverse=True))
# 4. Narysuj histogram (bar plot)
plt.figure(figsize=(14, 6))
plt.bar(sites, counts)
plt.xticks(rotation=90)
plt.ylabel("Liczba próbek")
plt.title("Rozkład primary sites w TCGA")
plt.tight_layout()
plt.show()
plt.savefig("primary_sites_distribution.png")