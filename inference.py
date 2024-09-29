import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块：Conv2d -> BatchNorm2d -> ReLU"""

    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConvBlock(nn.Module):
    """上采样块：ConvTranspose2d -> BatchNorm2d -> ReLU"""

    def __init__(self, in_channels, out_channels, stride=2):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, output_padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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
    """用于灰度图像上色的U-Net模型"""

    def __init__(self):
        super(ColorizationUNet, self).__init__()
        # 编码器
        self.enc1 = ConvBlock(1, 128)  # 输入为1通道灰度图像
        self.enc2 = ConvBlock(128, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 512)
        # 解码器
        self.dec1 = UpConvBlock(512, 512)
        self.dec2 = UpConvBlock(512 + 512, 256)  # 跳跃连接，通道数加倍
        self.dec3 = UpConvBlock(256 + 256, 128)
        self.dec4 = UpConvBlock(128 + 128, 128)
        self.dec5 = UpConvBlock(128 + 128, 3)  # 输出3通道彩色图像
        # 最后的深度可分离卷积层
        self.final_conv = SeparableConv2d(3 + 1, 3)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)  # 输出尺寸：(batch_size, 128, 80, 80)
        enc2 = self.enc2(enc1)  # 输出尺寸：(batch_size, 128, 40, 40)
        enc3 = self.enc3(enc2)  # 输出尺寸：(batch_size, 256, 20, 20)
        enc4 = self.enc4(enc3)  # 输出尺寸：(batch_size, 512, 10, 10)
        enc5 = self.enc5(enc4)  # 输出尺寸：(batch_size, 512, 5, 5)
        # 解码器路径
        dec1 = self.dec1(enc5)  # 输出尺寸：(batch_size, 512, 10, 10)
        dec1 = torch.cat((dec1, enc4), dim=1)  # 跳跃连接，通道数为1024
        dec2 = self.dec2(dec1)  # 输出尺寸：(batch_size, 256, 20, 20)
        dec2 = torch.cat((dec2, enc3), dim=1)  # 通道数为512
        dec3 = self.dec3(dec2)  # 输出尺寸：(batch_size, 128, 40, 40)
        dec3 = torch.cat((dec3, enc2), dim=1)  # 通道数为256
        dec4 = self.dec4(dec3)  # 输出尺寸：(batch_size, 128, 80, 80)
        dec4 = torch.cat((dec4, enc1), dim=1)  # 通道数为256
        dec5 = self.dec5(dec4)  # 输出尺寸：(batch_size, 3, 160, 160)
        # 拼接输入和解码器输出
        dec5 = torch.cat((dec5, x), dim=1)  # 通道数为4
        # 最后的卷积层
        out = self.final_conv(dec5)  # 输出尺寸：(batch_size, 3, 160, 160)
        out = self.sigmoid(out)  # 将输出限制在0到1之间
        return out


# 示例用法

model = ColorizationUNet()
# 创建一个示例灰度图像，尺寸为(1, 1, 160, 160)
input_image = torch.randn(1, 1, 160, 160)
# 前向传播
output_image = model(input_image)
print(output_image.shape)  # 输出尺寸应为(1, 3, 160, 160)

# 读取并预处理灰度图像
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

# 设置模型为评估模式
model.eval()

# 读取并预处理灰度图像
from PIL import Image
import torchvision.transforms as transforms
# 保存彩色图像
import matplotlib.pyplot as plt

# 定义与训练时相同的变换
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])
# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('./checkpoints/colorization_epoch_50.pth', map_location=device))
# 加载灰度图像并预处理
gray_image = Image.open('landscape/color/0.jpg').convert('L')
gray_tensor = transform(gray_image).unsqueeze(0).to(device)  # 添加批次维度并移动到设备

# 使用模型进行预测
with torch.no_grad():
    output_color = model(gray_tensor)

# 将输出转换为图像并保存
output_color = output_color.squeeze(0).cpu().numpy()  # 移除批次维度并移动到 CPU
output_img = np.transpose(output_color, (1, 2, 0))  # 调整维度顺序为 [H, W, C]

plt.imshow(output_img)
plt.imsave('c2.png', output_img)