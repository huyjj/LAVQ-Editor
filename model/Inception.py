from __future__ import division
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
'''-------------------------第一步：定义基础卷积模块-------------------------------'''
class BasicConv1d(nn.Module):
 
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
 
 
    '''-----------------第二步：定义Inceptionv3模块---------------------'''
 
'''---InceptionA---'''
class InceptionA(nn.Module):
 
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)
 
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)
 
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
 
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)
 
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
 
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
 
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
 
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
 
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs
 
    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
 
'''---InceptionB---'''
class InceptionB(nn.Module):
 
    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)
 
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)
 
    def _forward(self, x):
        branch3x3 = self.branch3x3(x)
 
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
 
        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)
 
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs
 
    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
 
'''---InceptionC---'''
class InceptionC(nn.Module):
 
    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)
 
        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=1, padding=0)
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=7, padding=3)
 
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=7, padding=3)
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=1, padding=0)
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=7, padding=3)
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=1, padding=0)
 
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
 
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
 
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
 
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
 
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
 
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs
 
    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
 
'''---InceptionD---'''
class InceptionD(nn.Module):
 
    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
 
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=1, padding=0)
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=7, padding=3)
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)
 
    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
 
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
 
        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs
 
    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
 
'''---InceptionE---'''
class InceptionE(nn.Module):
 
    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
 
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=1, padding=0)
        self.branch3x3_2b = conv_block(384, 384, kernel_size=3, padding=1)
 
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=1, padding=0)
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=3, padding=1)
 
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
 
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
 
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)
 
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
 
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
 
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs
 
    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
 
 
 
'''-------------------第三步：定义辅助分类器InceptionAux-----------------------'''
class InceptionAux(nn.Module):
 
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001
 
    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool1d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x
 
'''-----------------------第四步：搭建GoogLeNet网络--------------------------'''
class GoogLeNet(nn.Module):
 
    def __init__(self, num_classes=1000, in_c=12, aux_logits=False, transform_input=False,
                 inception_blocks=None):
        super(GoogLeNet, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv1d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
 
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv1d_1a_3x3 = conv_block(in_c, 32, kernel_size=3, stride=2)
        self.Conv1d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv1d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv1d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv1d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.fc = nn.Linear(2048, num_classes)
    '''输入(229,229,3)的数据，首先归一化输入，经过5个卷积，2个最大池化层。'''
    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv1d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv1d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv1d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv1d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv1d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        '''然后经过3个InceptionA结构，
        1个InceptionB，3个InceptionC，1个InceptionD，2个InceptionE，
        其中InceptionA，辅助分类器AuxLogits以经过最后一个InceptionC的输出为输入。'''
        # 35 x 35 x 192
        x = self.Mixed_5b(x)  # InceptionA(192, pool_features=32)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)  # InceptionA(256, pool_features=64)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)  # InceptionA(288, pool_features=64)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)  # InceptionB(288)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)  # InceptionC(768, channels_7x7=128)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)  # InceptionC(768, channels_7x7=160)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)  # InceptionC(768, channels_7x7=160)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)  # InceptionC(768, channels_7x7=192)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)  # InceptionAux(768, num_classes)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)  # InceptionD(768)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)  # InceptionE(1280)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)  # InceptionE(2048)
 
        '''进入分类部分。
        经过平均池化层+dropout+打平+全连接层输出'''
        x = F.adaptive_avg_pool1d(x, 1)
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)#Flatten（）就是将1d的特征图压扁为1D的特征向量，是展平操作，进入全连接层之前使用,类才能写进nn.Sequential
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x#, aux
 
    def forward(self, x):
        x = self._forward(x)
        return x#, aux
    '''-----------------------第五步：网络结构参数初始化--------------------------'''
 
    # 目的：使网络更好收敛，准确率更高
    def _initialize_weights(self):  # 将各种初始化方法定义为一个initialize_weights()的函数并在模型初始后进行使用。
 
        # 遍历网络中的每一层
        for m in self.modules():
            # isinstance(object, type)，如果指定的对象拥有指定的类型，则isinstance()函数返回True
 
            '''如果是卷积层Conv1d'''
            if isinstance(m, nn.Conv1d):
                # Kaiming正态分布方式的权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
                '''判断是否有偏置：'''
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # torch.nn.init.constant_(tensor, val)，初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
 
                '''如果是全连接层'''
            elif isinstance(m, nn.Linear):
                # init.normal_(tensor, mean=0.0, std=1.0)，使用从正态分布中提取的值填充输入张量
                # 参数：tensor：一个n维Tensor，mean：正态分布的平均值，std：正态分布的标准差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
'''---------------------------------------显示网络结构-------------------------------'''
if __name__ == '__main__':
    net = GoogLeNet(5, 12).cuda()
    from torchsummary import summary
    x = torch.randn((4, 12, 4096)).cuda()
    y= net(x)
    print(y)
    summary(net, (12, 4096))
 