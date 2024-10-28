import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Daten für Resilience
oneTree_success = [0.6, 1.0, 0.6, 0.9333]
oneTreeCP_success = [0.84, 0.96, 0.8, 0.8667]
x_values = [27, 24, 33, 59]

# Daten für Hops
oneTree_hops = [7.6, 4.8, 8.2, 5.67]
oneTreeCP_hops = [7.4, 3.8, 5.8, 19.33]

# DataFrame für Resilience
data_resilience = {
    'Graph Size (n)': x_values * 2,
    'Resilience (%)': oneTree_success + oneTreeCP_success,
    'Algorithm': ['OneTree'] * 3 + ['OneTreeCheckpoint'] * 3
}
df_resilience = pd.DataFrame(data_resilience)

# DataFrame für Hops
data_hops = {
    'Graph Size (n)': x_values * 2,
    'Hops': oneTree_hops + oneTreeCP_hops,
    'Algorithm': ['OneTree'] * 3 + ['OneTreeCheckpoint'] * 3
}
df_hops = pd.DataFrame(data_hops)

# Subplots erstellen
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot für Resilience erstellen mit Seaborn
sns.barplot(ax=axes[0], x='Graph Size (n)', y='Resilience (%)', hue='Algorithm', data=df_resilience, palette={'OneTree': 'green', 'OneTreeCheckpoint': 'red'})
axes[0].set_xlabel('Graph Size (n)')
axes[0].set_ylabel('Resilience (%)')
axes[0].set_title('Resilience of Algorithms for Topologies from Topology Zoo')

# Plot für Hops erstellen mit Seaborn
sns.barplot(ax=axes[1], x='Graph Size (n)', y='Hops', hue='Algorithm', data=df_hops, palette={'OneTree': 'green', 'OneTreeCheckpoint': 'red'})
axes[1].set_xlabel('Graph Size (n)')
axes[1].set_ylabel('Hops')
axes[1].set_title('Hops of Algorithms for Topologies from Topology Zoo')

# Layout anpassen und Plot anzeigen
plt.tight_layout()
plt.show()
