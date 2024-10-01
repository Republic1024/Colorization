import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """卷积块：Conv2d -> BatchNorm2d -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    """上采样块：Upsample -> Conv2d -> BatchNorm2d -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SeparableConv2d(nn.Module):
    """深度可分离卷积：Depthwise Conv2d + Pointwise Conv2d"""

    def __init__(self, in_channels, out_channels):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ColorizationUNet(nn.Module):
    """用于灰度图像上色的 U-Net 模型，支持任意大小的输入"""

    def __init__(self):
        super(ColorizationUNet, self).__init__()
        # 编码器
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 512)
        # 解码器
        self.dec1 = UpConvBlock(512, 512)
        self.dec2 = UpConvBlock(512 + 512, 256)
        self.dec3 = UpConvBlock(256 + 256, 128)
        self.dec4 = UpConvBlock(128 + 128, 64)
        self.dec5 = UpConvBlock(64 + 64, 3)
        # 最后的深度可分离卷积层
        self.final_conv = SeparableConv2d(3 + 1, 3)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        # 解码器路径
        dec1 = self.dec1(enc5)
        dec1 = torch.cat((dec1, enc4), dim=1)  # 跳跃连接
        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec4 = self.dec4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec5 = self.dec5(dec4)
        # 拼接输入和解码器输出
        dec5 = torch.cat((dec5, x), dim=1)
        # 最后的卷积层
        out = self.final_conv(dec5)
        out = self.sigmoid(out)
        return out


# 示例用法
if __name__ == "__main__":
    model = ColorizationUNet()
    # 创建一个示例灰度图像，尺寸为 (1, 1, 500, 1000)
    input_image = torch.randn(1, 1, 512, 512)
    output_image = model(input_image)
    print(output_image.shape)  # 应输出 (1, 3, 500, 1000)
