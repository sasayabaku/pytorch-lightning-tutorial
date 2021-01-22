import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet152

__all__ = [
    'ResNet18Transfer', 'ResNet18FineTuning',
    'ResNet152Transfer', 'ResNet152FineTuning'
    ]


class LightningBase(pl.LightningModule):
    def __init__(self):
        super(LightningBase, self).__init__()

        self.define_metrics()

        assert NotImplementedError

    def forward(self, *args, **kwargs):
        assert NotImplementedError

    def define_metrics(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        lambda_func = lambda epoch: 0.95 ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self.forward(data)

        loss = self.criterion(outputs, labels)
        self.train_acc(outputs, labels)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self.forward(data)

        loss = self.criterion(outputs, labels)
        self.valid_acc(outputs, labels)

        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        self.log('valid_acc', self.valid_acc.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)


class ResNet18FineTuning(LightningBase):
    def __init__(self, n_class):
        super().__init__()

        self.model = resnet18(pretrained=True)
        resnet_feature_extraction_output_dim = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(resnet_feature_extraction_output_dim, n_class)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


class ResNet18Transfer(LightningBase):
    def __init__(self, n_class):
        super().__init__()

        self.model = resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        resnet_feature_extraction_output_dim = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(resnet_feature_extraction_output_dim, n_class)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


class ResNet152FineTuning(LightningBase):
    def __init__(self, n_class):
        super().__init__()

        self.model = resnet152(pretrained=True)
        resnet_feature_extraction_output_dim = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(resnet_feature_extraction_output_dim, n_class)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


class ResNet152Transfer(LightningBase):
    def __init__(self, n_class):
        super().__init__()

        self.model = resnet152(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        resnet_feature_extraction_output_dim = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(resnet_feature_extraction_output_dim, n_class)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x
