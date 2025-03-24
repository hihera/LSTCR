import pandas as pd
import os

def convert_to_int_or_na(value):
    try:
        return int(value)  # 尝试转换为整数
    except ValueError:
        return value  # 如果转换失败，返回原始字符串

# 定义文件路径
input_base_dir = 'E:/workspace/thesis data/mall'
input_file_path = os.path.join(input_base_dir, 'customer_feature_input.txt')
output_file_path = os.path.join(input_base_dir, 'customer_embed_path_new.txt')

# 读取输入文件
data = pd.read_csv(
    input_file_path,
    sep='\t',
    dtype={'customer_id': str},  # 保持 customer_id 为字符串
    converters={'age': convert_to_int_or_na, 'gender': convert_to_int_or_na}  # 为age和gender指定转换器
)

# 输出读取后的数据类型和一些数据行以验证
#print(data.dtypes)
#print(data.head())

# 读取各属性的输出文件路径并保留完整信息
def get_embedding_info(feature_file, item):
    feature_path = os.path.join(input_base_dir, feature_file)
    if os.path.exists(feature_path):
        with open(feature_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                # 调试信息
                # print("=======parts========")
                # print(parts)
                # print("+++++++++++++item")
                # print(item)
                if str(parts[0]) == str(item):  # 确保比较时两者均为字符串
                    return line.strip()  # 返回完整的行信息
    return 'Item_Not_Found|NA|Path_Not_Found'

# 指定一个特定的 customer_id 进行调试
# specific_customer_id = '1451048123152465920'

# 生成新文件内容
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    for index, row in data.iterrows():
        # if row['customer_id'] == specific_customer_id:
        #     print(f"Processing customer_id: {specific_customer_id}")
        #     print(f"Age: {row['age']}, Gender: {row['gender']}")
        embed_infos = [
            get_embedding_info('customer_age_output_new.txt', row['age']),
            get_embedding_info('customer_gender_output_new.txt', row['gender']),
        ]

        line = f"{row['customer_id']}&" + '&'.join(embed_infos)
        out_file.write(line + '\n')

print("Processed and saved to", output_file_path)