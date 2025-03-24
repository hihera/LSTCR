import torch
import torch.nn as nn
import os
import csv

# 假设的文件路径和输出目录
input_file_path = 'E:/workspace/thesis data/mall/context_feature_input.txt'
output_base_dir = 'E:/workspace/thesis data/mall'

# 定义需要处理的特征及其embedding维度
features = {
    'is_holiday': 2,
    'month': 12,
    'season': 4,
    'weekday': 7,
    'time_of_day': 4,
    'time_interval': 48  # 假设一天按30分钟间隔，共48个时间段
}

# 读取数据并处理
features_indices = {feature: {} for feature in features}
with open(input_file_path, 'r', encoding='utf-8', newline='') as file:
    reader = csv.reader(file, delimiter='\t', quotechar='"')  # 使用制表符作为分隔符，并处理引号
    headers = next(reader)  # 读取标题行
    for data in reader:
        for i, header in enumerate(headers):
            if header in features:
                item = data[i].strip()
                if item not in features_indices[header]:
                    features_indices[header][item] = len(features_indices[header])

# 创建embedding层
embeddings = {}
for feature, dim in features.items():
    num_items = len(features_indices[feature])
    embeddings[feature] = nn.Embedding(num_items, dim)

# 初始化随机种子以保证结果可复现
torch.manual_seed(0)

# 保存embeddings并写入输出文件
for feature, embedding_layer in embeddings.items():
    feature_embed_dir = os.path.join(output_base_dir, f'context_{feature}_embed')
    os.makedirs(feature_embed_dir, exist_ok=True)  # 确保每个属性的embedding存储目录存在
    feature_output_path = os.path.join(output_base_dir, f'context_{feature}_output.txt')
    with open(feature_output_path, 'w', encoding='utf-8') as output_file:
        #output_file.write("item|index|embedding_path\n")
        for item, index in features_indices[feature].items():
            # 提取embedding
            item_embedding = embedding_layer(torch.tensor([index])).squeeze(0).detach()

            # 保存embedding
            embedding_path = os.path.join(feature_embed_dir, f'{index}_embedding.pt')
            torch.save(item_embedding, embedding_path)

            # 写入输出文件
            output_file.write(f'{item}|{index}|{embedding_path}\n')

print("Preprocessing completed. Embeddings for each feature are saved and indexed separately.")