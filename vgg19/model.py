import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import vgg19


class Vgg19Base(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.vgg19 = vgg19(pretrained=True)

        self.define_metrics()

        assert NotImplementedError

    def forward(self, inputs):
        assert NotImplementedError

    def define_metrics(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
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


class VGG19Transfer(Vgg19Base):

    def __init__(self, n_class):
        super().__init__()

        for param in self.vgg19.parameters():
            param.requires_grad = False

        vgg_feature_extract_output_dim = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = torch.nn.Linear(vgg_feature_extract_output_dim, n_class)

        self.model = self.vgg19

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


class VGG19FineTuning(Vgg19Base):

    def __init__(self, n_class):
        super().__init__()

        vgg_feature_extract_output_dim = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = torch.nn.Linear(vgg_feature_extract_output_dim, n_class)

        self.model = self.vgg19

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9)

        return optimizer
