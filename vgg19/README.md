# 学習済みモデルの活用

<a href="https://www.kaggle.com/sasayabaku/vgg19-cifar10-transfer-learning-fine-tuning" target="_blank" rel="noreferrer">
    <img src="https://img.shields.io/badge/kaggle-Notebook-blue?logo=kaggle" />
</a>

VGG19にCIFAR-10を再学習させるNotebook  

- `Vgg19Base` : 共通の学習用メソッド
- `VGG19Transfer` : Transfer Learning 用 Class
- `VGG19FineTuning` : Fine Tuning 用 Class

# 各種Point

## Fine-Tuning

### 学習済みの重みの固定

In `model.py`  
`VGG19Transferクラス`内の

```python
for param in self.vgg19.parameters():
    param.requires_grad = False
```
にて、Pretrainedモデルの重みを固定する

## Transfer Learning (転移学習)

### 学習率の修正

In `VGG19Base`  

デフォルトの最適化関数を下記に設定していると
```python
optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
```

学習率が大きすぎるからか、学習中のlossがnanになってしまうため、  
`lr=0.005`に修正する。
