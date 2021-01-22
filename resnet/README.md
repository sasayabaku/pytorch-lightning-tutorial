# 学習済みモデルの活用

<a href="https://www.kaggle.com/sasayabaku/resnet-cifar10-transfer-learning-fine-tuning" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/kaggle-Notebook-blue?logo=kaggle" />
</a>

ResNetにCIFAR-10を再学習させるNotebook   
(CPU / GPU学習想定)

- `LightningBase` : 共通の学習用メソッド
- `ResNet18Transfer` : ResNet-18 Transfer Learning
- `ResNet19FineTuning` : ResNet-18 Fine Tuning
- `ResNet152Transfer` : ResNet-152 Transfer Learning
- `ResNet152FineTuning` : ResNet-152 Fine Tuning

# 各種Point

## Fine-Tuning

### 学習済みの重みの固定

In `model.py`  
`ResNet18FineTuning / ResNet152FineTuning クラス`内の

```python
for param in self.model.parameters():
    param.requires_grad = False
```
にて、Pretrainedモデルの重みを固定する