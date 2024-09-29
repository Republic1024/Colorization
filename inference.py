import torch
import torch.nn as nn
import torch.nn.functional as F
from colorization_unet import ColorizationUNet

model = ColorizationUNet()
# 创建一个示例灰度图像，尺寸为(1, 1, 160, 160)
input_image = torch.randn(1, 1, 160, 160)
# 前向传播
output_image = model(input_image)
print(output_image.shape)  # 输出尺寸应为(1, 3, 160, 160)

# 读取并预处理灰度图像
from PIL import Image
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
model.load_state_dict(torch.load('./checkpoints/colorization_epoch_50.pth', map_location=device,weights_only=True))
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
plt.imsave('./output/0_colorized.png', output_img)
