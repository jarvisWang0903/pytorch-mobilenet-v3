import torch
import torch.nn.functional as F
import torch.nn as nn

class Hswish(nn.Module):
    def __init__(self,inpalce):
        super(Hswish,self).__init__()
        self.inplace = inpalce
    def forward(self, x):
        return x*F.relu6(x+3, inplace=self.inplace)/6.

class Hsigmoid(nn.Module):
    def __init__(self,inpalce):
        super(Hsigmoid,self).__init__()
        self.inplace = inpalce
    def forward(self, x):
        return F.relu6(x+3, inplace=self.inplace)/6.

class Squeeze_and_Excite(nn.Module):
    def __init__(self,inplane):
        super(Squeeze_and_Excite,self).__init__()
        midplane = inplane // 16
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.FC1 = nn.Linear(inplane, midplane,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.FC2 = nn.Linear(midplane, inplane,bias=True)
        self.hsigmoid = Hsigmoid(inpalce=True)
    def forward(self, x):
        b, c, _, _ = x.size()
        pool = self.globalpool(x).view(b, c)
        out = self.relu(self.FC1(pool))
        out = self.hsigmoid(self.FC2(out)).view(b, c, 1, 1)
        return out



class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplans, kernel, expantion,
                 stride, SE=False, nl='HS'):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        if nl == 'HS':
            self.non_linearity = Hswish(True)
        elif nl == 'RE':
            self.non_linearity = nn.ReLU(True)
        else:
            raise NotImplementedError('this non_linearity method not implemented')
        self.SE = SE
        self.SEmodel = Squeeze_and_Excite(expantion)
        self.use_res_connect = stride == 1 and inplanes == outplans
        self.conv1 = nn.Conv2d(inplanes, expantion, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expantion)
        self.relu = nn.ReLU()
        self.dwiseConv = nn.Conv2d(expantion, expantion, kernel_size=kernel, stride=stride,
                                   padding=padding, groups=expantion, bias=False)
        self.bn2 = nn.BatchNorm2d(expantion)
        self.pointConv = nn.Conv2d(expantion,outplans, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplans)
    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.non_linearity(self.bn2(self.dwiseConv(out)))
        if self.SE:
            y = self.SEmodel(out)
            out = out * y.expand_as(out)
        out = self.non_linearity(self.bn3(self.pointConv(out)))
        if self.use_res_connect:
            out = x + out

        return out






class MobileNet(nn.Module):
    def __init__(self, class_num, mode, width_mult=1.0):
        super(MobileNet,self).__init__()
        self.mode = mode.lower()
        if mode.lower() == 'small':
            setting = [
                # k,exp,outp,SE,  NL,  s
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
        elif mode.lower() == 'large':
            setting = [
                # k,exp,inp,outp,  se,   nl,  s,
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
            ]
        else:
            raise NotImplementedError
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(16),
                                   Hswish(inpalce=True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  Hswish(inpalce=True))
        self.bottlenet = nn.Sequential()
        self.conv_last0 = nn.Sequential(nn.Conv2d(96 if mode.lower()=='small' else 160,
                                                  576 if mode.lower()=='small' else 960,
                                                  kernel_size=1,stride=1,padding=0,bias=False),
                                        nn.BatchNorm2d(576 if mode.lower()=='small' else 960),
                                        Hswish(inpalce=True))
        inp = 16
        for idx, param in enumerate(setting):
            [k, exp, outp, se, nl, s] = param
            self.bottlenet.add_module('bottleneck' + str(idx), Bottleneck(inp, outp, k, exp, s, se, nl))
            inp = outp
        self.conv_last1 = nn.Conv2d(960 if mode.lower() == 'large' else 576,
                                    1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_last2 = nn.Conv2d(1280, class_num, kernel_size=1, padding=0,bias=False)
        self.bn_last1 = nn.BatchNorm2d(1280)
        self.bn_last2 = nn.BatchNorm2d(class_num)
        self.hswish = Hswish(inpalce=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bottlenet(out)
        out = self.pool(self.conv_last0(out))
        if self.mode == 'large':
            out = self.conv_last2(self.hswish(self.conv_last1(out)))
        elif self.mode == 'small':
            out = self.hswish(self.bn_last1(self.conv_last1(out)))
            out = self.hswish(self.bn_last2(self.conv_last2(out)))
        else:
            raise NotImplementedError
        return out


if __name__ == '__main__':
    net = MobileNet(21,'large')
    x = torch.randn((4, 3, 224, 224))
    out = net(x)
    print(out.size())