import torch
from torch import nn
from fastai.vision.models import resnet18
from functools import partial
from fastai.callback import Callback
from fastai.layers import AdaptiveConcatPool2d, relu, Flatten, bn_drop_lin


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.5):
        super().__init__()
        layers = [AdaptiveConcatPool2d(), relu(), Flatten()] + \
            bn_drop_lin(nc*2, 512, True, ps, relu()) + \
            bn_drop_lin(512, n, True, ps)
        self.fc = nn.Sequential(*layers)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


# change the first conv to accept 1 chanel input
class Rnet_1ch(nn.Module):
    def __init__(self, arch=resnet18, n=[168, 11, 7], pre=True, ps=0.5):
        super().__init__()
        m = arch(True) if pre else arch()

        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        w = (m.conv1.weight.sum(1)).unsqueeze(1)
        conv.weight = nn.Parameter(w)

        self.layer0 = nn.Sequential(conv, m.bn1, nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1,
                         dilation=1,
                         ceil_mode=False), m.layer1)
        self.layer2 = nn.Sequential(m.layer2)
        self.layer3 = nn.Sequential(m.layer3)
        self.layer4 = nn.Sequential(m.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # m.avgpool

        # nc = self.layer4[-1].weight.shape[0]
        nc = 512
        self.head1 = Head(nc, n[0])
        self.head2 = Head(nc, n[1])
        self.head3 = Head(nc, n[2])

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0),-1)
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return x1, x2, x3


class Metric_idx(Callback):
    def __init__(self, idx, average='macro'):
        super().__init__()
        self.idx = idx
        self.n_classes = 0
        self.average = average
        self.cm = None
        self.eps = 1e-9

    def on_epoch_begin(self, **kwargs):
        self.tp = 0
        self.fp = 0
        self.cm = None

    def on_batch_end(self, last_output: torch.Tensor,
                     last_target: torch.Tensor, **kwargs):
        last_output = last_output[self.idx]
        last_target = last_target[:, self.idx]
        preds = last_output.argmax(-1).view(-1).cpu()
        targs = last_target.long().cpu()

        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
            self.x = torch.arange(0, self.n_classes)
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])) \
          .sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm = cm
        else: self.cm += cm

    def _weights(self, avg: str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn("average=`binary` was selected for a non binary case. \
                 Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1:
                return Tensor([0, 1])
            else:
                return Tensor([1, 0])
        elif avg == "micro":
            return self.cm.sum(dim=0) / self.cm.sum()
        elif avg == "macro":
            return torch.ones((self.n_classes, )) / self.n_classes
        elif avg == "weighted":
            return self.cm.sum(dim=1) / self.cm.sum()

    def _recall(self):
        rec = torch.diag(self.cm) / (self.cm.sum(dim=1) + self.eps)
        if self.average is None: return rec
        else:
            if self.average == "micro": weights = self._weights(avg="weighted")
            else: weights = self._weights(avg=self.average)
            return (rec * weights).sum()

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._recall())


Metric_grapheme = partial(Metric_idx, 0)
Metric_vowel = partial(Metric_idx, 1)
Metric_consonant = partial(Metric_idx, 2)


class Metric_tot(Callback):
    def __init__(self):
        super().__init__()
        self.grapheme = Metric_idx(0)
        self.vowel = Metric_idx(1)
        self.consonant = Metric_idx(2)

    def on_epoch_begin(self, **kwargs):
        self.grapheme.on_epoch_begin(**kwargs)
        self.vowel.on_epoch_begin(**kwargs)
        self.consonant.on_epoch_begin(**kwargs)

    def on_batch_end(self, last_output: torch.Tensor,
                     last_target: torch.Tensor, **kwargs):
        self.grapheme.on_batch_end(last_output, last_target, **kwargs)
        self.vowel.on_batch_end(last_output, last_target, **kwargs)
        self.consonant.on_batch_end(last_output, last_target, **kwargs)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(
            last_metrics, 0.5 * self.grapheme._recall() +
            0.25 * self.vowel._recall() + 0.25 * self.consonant._recall())
