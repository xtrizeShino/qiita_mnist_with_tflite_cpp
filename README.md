# TensorFlow Lite for C/C++で推論するサンプルプロジェクト

for Qiita "[Ubuntu22.04上でTensorFlow Lite for C/C++をソースコードからx86_64(CPU)向けにビルドして、Native-Linux環境で実装したC++のプログラムからTensorFlow Liteを呼び出す](https://qiita.com/xtrizeShino/items/4d521ebf06a3f40a9490)"

## ソースコードの入手元
- [Deep Learningアプリケーション開発 (1)](https://qiita.com/iwatake2222/items/796ec8560563625ace34)
- [Deep Learningアプリケーション開発 (4)](https://qiita.com/iwatake2222/items/d63aa67e5c700fcea70a)
- [Deep Learningアプリケーション開発 (5)](https://qiita.com/iwatake2222/items/d998df1981d46285df62)

## 利用方法

詳細は "[Ubuntu22.04上でTensorFlow Lite for C/C++をソースコードからx86_64(CPU)向けにビルドして、Native-Linux環境で実装したC++のプログラムからTensorFlow Liteを呼び出す](https://qiita.com/xtrizeShino/items/4d521ebf06a3f40a9490)" の記事をご参照ください。

### TensorFlow for C/C++ 

GUIのある環境で以下を実行してください

```bash
### ソースコードを入手する
$ git clone [This Repository]
$ cd [This Repository]
### ビルドディレクトリを作成する
$ mkdir build
$ cd build
### ビルドする
$ cmake ..
$ make
### 実行
$ ./NumberDetector
```

### Generate TFLite (by Python)

GUIのある環境で以下を実行してください

```bash
### モデルの定義と学習
python mnist_train.py
### 学習結果を利用した推論
python mnist_infer.py
### TensorFlow -> TensorFlow Lite
python mnist_convert.py
### TensorFlow Liteで推論
python mnist_infer_tflite.py
```
