# KÜMELEME-YAPAY-ZEKA
Iris veri seti üzerinde K-Means kümeleme algoritmasını kullanarak veri noktalarını kümelere ayırma ve çeşitli analizler yapma işlemlerini gerçekleştirir. 


1. Veri Yükleme ve İşleme
Iris Veri Setini Yükleme:
datasets.load_iris() fonksiyonu ile Iris veri seti yüklenir. Bu veri seti, iris çiçeğinin özelliklerini içeren bir veri kümesidir ve genellikle sınıflandırma ve kümeleme problemleri için kullanılır.


2.Veri Çerçevesi Oluşturma:
Veriler bir pandas DataFrame'ine dönüştürülür. Özellikler (canak uzunluğu, canak genişliği, taç uzunluğu, taç genişliği) ve tür bilgisi DataFrame'de saklanır.


3. Veri Analizi ve Görselleştirme
Korelasyon Isı Haritası:
Özellikler arasındaki korelasyon hesaplanır ve bir ısı haritası (heatmap) ile görselleştirilir. Bu, özelliklerin birbirleri ile ne kadar ilişkili olduğunu anlamanıza yardımcı olur.
Saçılım Diyagramları:

Özellik çiftleri arasındaki ilişkiler, tür bilgisi ile renklendirilmiş saçılım diyagramları ile görselleştirilir. Bu, veri setindeki niteliklerin nasıl dağıldığını ve kümelendiğini görsel olarak incelemenizi sağlar.
3. K-Means Kümeleme
K-Means Kümeleme Modeli:

KMeans algoritması kullanılarak veri noktaları kümelere ayrılır. fit_predict yöntemi, veri noktalarını kümelere atar ve her bir veri noktasının hangi kümeye ait olduğunu belirler.
Kümelerin Görselleştirilmesi:

Kümeler ve küme merkezleri, 2D grafikte renkli noktalarla görselleştirilir. Kümeler farklı renklerde gösterilir ve küme merkezleri büyük bir işaret ile gösterilir.
4. Kümeleme Kalitesinin Ölçülmesi
Silhouette Skoru:

silhouette_score fonksiyonu, kümelerin ne kadar iyi ayrıldığını ölçmek için kullanılır. Silhouette skoru, kümeler arasındaki uzaklığı ve kümeler içindeki yoğunluğu değerlendirir. Skor ne kadar yüksekse, kümeler o kadar iyi ayrılmış demektir.
Silhouette Visualizer:

SilhouetteVisualizer kullanılarak kümelenmenin görsel bir değerlendirmesi yapılır. Bu, her bir veri noktasının ne kadar iyi bir şekilde kümelendiğini gösterir.
5. En İyi Küme Sayısını Belirleme
Elbow Yöntemi:

Küme sayısına bağlı olarak WCSS (Within-Cluster Sum of Squares) hesaplanır ve bir grafik üzerinde görselleştirilir. Bu grafik, en uygun küme sayısını belirlemenize yardımcı olabilir. "Dirsek" (elbow) noktası, optimal küme sayısını gösterir.
Silhouette İndeks Değeri:

Farklı küme sayıları için silhouette skorları hesaplanır ve bir grafik üzerinde görselleştirilir. Bu grafik, hangi küme sayısının en iyi olduğunu belirlemenize yardımcı olabilir.
