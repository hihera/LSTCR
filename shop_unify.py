import pandas as pd
import os

# 文件路径
input_file_path = 'E:/workspace/thesis data/mall/shop_feature_input.txt'
output_base_dir = 'E:/workspace/thesis data/mall'
output_file_path = os.path.join(output_base_dir, 'shop_embed_path.txt')

# 读取输入文件
data = pd.read_csv(input_file_path, sep='\t', quotechar='"')

# 读取各属性的输出文件路径并保留完整信息
def get_embedding_info(feature_file, item):
    feature_path = os.path.join(output_base_dir, feature_file)
    if os.path.exists(feature_path):
        with open(feature_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                if parts[0] == item:
                    return line.strip()  # 返回完整的行信息
    return 'Item_Not_Found|NA|Path_Not_Found'

# 归一化坐标
data['coordinate_x'] = (data['coordinate_x'] - data['coordinate_x'].min()) / (data['coordinate_x'].max() - data['coordinate_x'].min())
data['coordinate_y'] = (data['coordinate_y'] - data['coordinate_y'].min()) / (data['coordinate_y'].max() - data['coordinate_y'].min())

# 生成新文件内容
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    for index, row in data.iterrows():
        embed_infos = [
            get_embedding_info('shop_tenant_name_output.txt', row['tenant_name']),
            get_embedding_info('shop_sys_floor_output.txt', row['sys_floor']),
            get_embedding_info('shop_category_output.txt', row['category']),
            get_embedding_info('shop_zone_location_output.txt', row['zone_location'])
        ]
        #line = '&'.join(embed_infos + [str(row['coordinate_x']), str(row['coordinate_y'])])
        line = f"{row['tenant_name']}&" + '&'.join(embed_infos + [str(row['coordinate_x']), str(row['coordinate_y'])])

        out_file.write(line + '\n')

print("Processed and saved to", output_file_path)