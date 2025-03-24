import torch
import torch.nn as nn
import os
import csv
import  pandas as pd

# 假设的文件路径和输出目录
ori_file_path = 'E:/workspace/thesis data/mall/customer_feature.txt'
input_file_path = 'E:/workspace/thesis data/mall/customer_feature_input.txt'
output_base_dir = 'E:/workspace/thesis data/mall'

# 指定数据类型，确保gender以整数形式读入
dtype_dict = {
    'customer_id': str,
    'age': pd.Int64Dtype(),  # 使用 pandas 的 nullable integer type
    'gender': pd.Int64Dtype()  # 同上，保证读入为整数，同时支持NA
}

# 读取输入文件，并指定数据类型
data = pd.read_csv(ori_file_path, sep='\t', quotechar='"', dtype=dtype_dict)

data['age'] = data['age'].apply(lambda x: 'NA' if x == 0 else x)

# 处理 gender 列：确保 gender 只有 0 和 1，其他转为 'NA'
def convert_gender(gender):
    if pd.isna(gender):  # 检查是否为NA
        return 'NA'
    elif gender == 0:
        return 0
    elif gender == 1:
        return 1
    else:
        return 'NA'

data['gender'] = data['gender'].apply(convert_gender)

# 保存输出文件
data.to_csv(input_file_path, sep='\t', index=False, quotechar='"')

print("Processed and saved to", input_file_path)

# 定义需要处理的特征及其embedding维度
features = {
    'age': 16,
    'gender': 4
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
    feature_embed_dir = os.path.join(output_base_dir, f'customer_{feature}_embed_new')
    os.makedirs(feature_embed_dir, exist_ok=True)  # 确保每个属性的embedding存储目录存在
    feature_output_path = os.path.join(output_base_dir, f'customer_{feature}_output_new.txt')
    with open(feature_output_path, 'w', encoding='utf-8') as output_file:
        #output_file.write("item|index|embedding_path\n")
        for item, index in features_indices[feature].items():
            # 提取embedding
            item_embedding = embedding_layer(torch.tensor([index])).squeeze(0).detach()

            # 保存embedding
            embedding_path = os.path.join(feature_embed_dir, f'{index}_embedding_new.pt')
            torch.save(item_embedding, embedding_path)

            # 写入输出文件
            output_file.write(f'{item}|{index}|{embedding_path}\n')

print("Preprocessing completed. Embeddings for each feature are saved and indexed separately.")