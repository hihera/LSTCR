# 项目名称

## 简介
本项目基于 **PyTorch** 框架，使用深度学习方法进行数据分析和模型训练，支持 GPU 加速，适用于 Windows 10 及以上环境。
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.2.1%2Bcu121-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 环境要求
- **操作系统**: Windows 10
- **深度学习框架**: PyTorch 2.2.1 (CUDA 12.4, Cudnn 9.0)
- **完整依赖**: 详见 [`requirements.txt`](requirements.txt)
- **硬件要求**: NVIDIA GPU（至少 16GB 显存）

## 安装指南
推荐使用 Conda 创建虚拟环境并安装依赖：
```bash
# 创建并激活 Python 虚拟环境
conda create -n PyTest python=3.11.7 -y
conda activate PyTest

# 安装依赖
pip install -r requirements.txt
```

## 主要依赖
以下是 `requirements.txt` 文件中关键的依赖项：
```bash
torch==2.2.1
pytorch-cuda==12.1
transformers==4.40.0
accelerate==0.29.3
numpy==1.26.4
pandas==2.1.4
pyyaml==6.0.1
scipy==1.12.0
tokenizers==0.19.1
sentence-transformers==2.7.0
peft==0.10.0
timm==0.9.16
tqdm==4.66.2
requests==2.31.0
pytorch-geometric==2.5.2
scikit-learn==1.2.2
matplotlib==3.8.4
spyder==5.5.5
```

## 数据准备
1. 解压 `dataset/mall` 目录下的 mall.zip 文件：
   ```
   dataset/mall/
   ├── customer_feature.txt
   ├── context_feature.txt
   ├── customer_shop.txt
   └── shop_feature_input.txt
   ```
2. **运行数据预处理脚本**：
   ```bash
   python main.py
   ```
3. **开始训练模型**：
   ```bash
   python train_model.py
   ```

## 训练参数
训练脚本支持以下参数，用户可根据需求调整：
| 参数           | 说明              | 默认值  |
|--------------|-----------------|------|
| `--batch_size` | 批大小          | 32   |
| `--lr`        | 学习率          | 0.001 |
| `--model_dim` | 模型维度         | 256  |
| `--num_heads` | 多头注意力机制数 | 8    |
| `--num_layers` | Transformer 层数 | 3    |

示例运行：
```bash
python train_model.py --batch_size 32 --lr 0.001
```

## 结果评估
模型训练完成后，控制台会输出各个指标的评价结果。

## 许可证
本项目遵循 **MIT License** 许可证，详细信息请查看 [LICENSE](LICENSE) 文件。
