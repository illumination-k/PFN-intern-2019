# readme.md

## 課題について

### srcの内部

|ファイル名|説明|
|---|---|
|model.py|GNNクラスが定義されたファイル|
|function.py|各種関数が定義されたファイル|
|optimizer.py|各種最適化法のクラスが定義されたファイル|
|Task\_n(\_explanation).ipynb|課題nに対応したjupyter notebookファイル。説明があれば課題のうち特にそのことについてテストしたファイル|
|best\_model\_x.txt/ best\_model\_x.npz|学習した際に最もaccuracyが高かったモデルのT,D/学習器のデータが格納されている。|

#### 補足
GNNクラスはattributeとしてload機能を持ち、best\_model\_x.txtとbes\_model\_x.npzが存在するディレクトリパスとbest\_model\_xのファイル名を指定すれば学習器θの情報がロードできる。

### testについて
jupyter notebookファイルを上から直接実行すればほとんどのテストが実行できる。
また、学習結果は保存されているので、モデルをloadすることでmodelのaccuracyなどはある程度確認できると思われる。