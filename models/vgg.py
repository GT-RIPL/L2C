import torch.nn as nn

class VGG(nn.Module):
    cfg = {
        'S': [64, 'M', 128, 'M', 256, 'M', 256, 'M'],
        8: [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
        11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, n_layer, out_dim=10, in_channels=3, img_sz=32):
        super(VGG, self).__init__()
        self.conv_func = nn.Conv2d
        self.features = self._make_layers(VGG.cfg[n_layer],in_channels)
        if n_layer=='S':
            self.feat_map_sz = img_sz // 16
            feat_dim = 256*(self.feat_map_sz**2)
            self.last = nn.Sequential(
                nn.Linear(feat_dim, feat_dim//2),
                nn.BatchNorm1d(feat_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim//2, out_dim)
            )
            self.last.in_features = feat_dim
        else:
            self.feat_map_sz = img_sz // 32
            self.last = nn.Linear(512*(self.feat_map_sz**2), out_dim)

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x.view(x.size(0), -1))
        return x


def VGGS(out_dim):
    return VGG(n_layer='S', out_dim=out_dim, in_channels=1)


def VGG8(out_dim):
    return VGG(n_layer=8, out_dim=out_dim)


def VGG11(out_dim):
    return VGG(n_layer=11, out_dim=out_dim)


def VGG13(out_dim):
    return VGG(n_layer=13, out_dim=out_dim)


def VGG16(out_dim):
    return VGG(n_layer=16, out_dim=out_dim)


def VGG19(out_dim):
    return VGG(n_layer=19, out_dim=out_dim)
