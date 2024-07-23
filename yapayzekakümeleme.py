import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer

# 1. Iris Veri Setini Yükleme
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Veri Setinin DataFrame'de Saklanması
df = pd.DataFrame(X, columns=["canakU", "canakG", "tacU", "tacG"])
df["tur"] = y
df["tur"] = df["tur"].replace([0, 1, 2], list(iris.target_names))

# Korelasyon
my_cors = np.corrcoef(X, rowvar=False).round(2)
plt.figure(figsize=(10, 8))
sns.heatmap(my_cors, annot=True, square=True, cmap=sns.color_palette("flare", as_cmap=True))
plt.title("Korelasyon Isı Haritası")
plt.show()

# Yüksek ve Düşük Korelasyonlu Niteliklerin Saçılım Diyagramı
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x="canakU", y="canakG", hue="tur", data=df, palette="rainbow")
plt.title("canakU vs canakG")

plt.subplot(1, 2, 2)
sns.scatterplot(x="tacU", y="tacG", hue="tur", data=df, palette="rainbow")
plt.title("tacU vs tacG")
plt.show()

# 2. K-Ortalamalar (KMeans) Algoritması
kumeSayisi = 3
kOrtModeli = KMeans(n_clusters=kumeSayisi, init="k-means++", n_init=10, random_state=0)
labels = kOrtModeli.fit_predict(X)

# Kümeleri DataFrame'e Ekle
df["kumeler"] = labels
df["kumeler"] = df["kumeler"].astype("category")
print("Kümeler Frekansları:\n", df["kumeler"].value_counts())

# Kume Merkezleri
plt.figure(figsize=(8, 6))
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=50, c="red", label="Küme 0")
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=50, c="blue", label="Küme 1")
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s=50, c="green", label="Küme 2")
plt.scatter(kOrtModeli.cluster_centers_[:, 0], kOrtModeli.cluster_centers_[:, 1], s=100, c="black", label="Küme Merkezleri", marker='X')
plt.title("K-Ortalamalar ile Elde Edilen Kümeler")
plt.xlabel("canakU")
plt.ylabel("canakG")
plt.legend()
plt.show()

# Gerçek ve Tahmin Edilen Kümeleri Görselleştirme
plt.figure(figsize=(16, 8))
fig, axes = plt.subplots(1, 2)
sns.scatterplot(x="canakU", y="tacU", hue="tur", data=df, palette="rainbow", ax=axes[0]).set(title="Gerçek Kategoriler")
sns.scatterplot(x="canakU", y="tacU", hue="kumeler", data=df, palette="rainbow", ax=axes[1]).set(title="K-Ortalamalardan Elde Edilen Kümeler")
plt.show()

# Kümelenme Kalitesinin Bulunması (Silhouette Skoru)
silhouette_avg = silhouette_score(X, labels, metric="euclidean")
print(f"Silhouette İndeks Değeri: {silhouette_avg:.3f}")

# Silhouette Visualizer
visualizer = SilhouetteVisualizer(kOrtModeli, colors="yellowbrick")
visualizer.fit(X)
visualizer.show()

# Birden Fazla Küme Sayısını Deneme ve En İyi Küme Sayısını Bulma
# I. YOL: Elbow Yöntemi
wcss = []
k = range(2, 21)
for i in k:
    kOrt = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=0)
    kOrt.fit(X)
    wcss.append(kOrt.inertia_)

plt.figure(figsize=(12, 6))
plt.plot(k, wcss, 'bx-')
plt.xticks(k)
plt.title("Elbow Yöntemi ile En İyi Küme Sayısı")
plt.xlabel("Küme Sayısı")
plt.ylabel("WCSS")
plt.show()

# II. YOL: Silhouette İndeks Değeri
silhs = []
for i in k:
    kOrt = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=0)
    kOrt.fit(X)
    silh_degeri = silhouette_score(X, kOrt.labels_, metric="euclidean")
    silhs.append(silh_degeri)

plt.figure(figsize=(12, 6))
plt.plot(k, silhs, 'bx-')
plt.xticks(k)
plt.title("Silhouette İndeks Değeri ile En İyi Küme Sayısı")
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette")
plt.show()
