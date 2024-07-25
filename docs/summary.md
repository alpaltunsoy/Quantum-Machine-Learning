# Quantum Computing
Bu yazıda genel hatlarıyla kodların açıklaması olacak Standart bir makine öğrenmesinin detayları yer almayacak sadece yazılmış QUANTUM VARIATIONAL CLASSIFIER (QVC) detayları olacak. Yazı sonunda ise bazı analizler olacak. Temel bilgileri edinmek için pdf'i okuyabilirsiniz.
## QVC NEDİR



## Kodların Detayları
Kuantum hesaplama ve sınıflandırma yapabilmek için öncelikle kesinlikle ön işleme(pre-processing ) yapılmalıdır. Bu kısımda herhangi bir makine öğrenmesinde kullanılan tüm ön işlemeler yapılmıştır. Scaling, Data Seperation vs. ilgili kodların fonksiyon başlıkları aşağıda yer almaktadır

`load_data()`
`minimize_rows(data, index,name)`
`encoding_and_creating_sets_for_x(data)`
`make_y(y)`
`missing_values(x)`
`scaling_data(x)`
`y_encoding(y)`

## QVC Fonksiyonu
### QVC NASIL ÇALIŞIR

```
def quantum_algorithm(x_train, y_train, x_test, y_test):
    print("Quantum algorithm is started")
    global num_features
    print("Finding features")

```

`num_features` Eğitim veri kümesindeki özellik sayısı (kolon sayısı), kuantum devrelerinin boyutunu tanımlamak için kullanılır.

```
num_features = x_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", style="clifford", fold=20)

```

Bir `ZZFeatureMap`, klasik verileri kuantum özellik uzayına eşlemek için kullanılır. `reps` parametresi, özellik haritasının kaç kez tekrar edileceğini belirler.

```
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
```


`RealAmplitudes` ansatz, kuantum durumunu parametreleştiren kuantum devresini oluşturmak için kullanılır. `reps` parametresi, ansatz devresinin kaç kez tekrar edileceğini belirtir.