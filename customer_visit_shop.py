import pandas as pd
import os

# 定义文件路径
input_base_dir = 'E:/workspace/thesis data/mall'
shop_input_file_path = os.path.join(input_base_dir, 'customer_shop_latest5.txt')
# shop_input_file_path = os.path.join(input_base_dir, 'customer_shop.txt')
output_file_path = os.path.join(input_base_dir, 'customer_shop_input.txt')

# 读取数据
shop_data = pd.read_csv(shop_input_file_path, sep='\t')

# 格式化日期为 YYYY-MM-DD
# shop_data['capture_date'] = pd.to_datetime(shop_data['capture_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
shop_data['capture_date'] = pd.to_datetime(shop_data['capture_date'], errors='coerce').dt.strftime('%Y-%m-%d')
if shop_data['capture_date'].isnull().any():
    print("Some dates could not be converted. Please check the date formats.")

# 合并相同 customer_id 的记录
merged_data = shop_data.groupby('customer_id').apply(lambda x: x.to_dict(orient='records')).reset_index()
merged_data.columns = ['customer_id', 'records']

# 格式化数据
formatted_data = []
for _, row in merged_data.iterrows():
    customer_id = row['customer_id']
    records = row['records']
    formatted_records = [
        [record['capture_date'], record['entry_time'], record['tenant_name'], record['diff_sec']]
        for record in records
    ]
    formatted_data.append([customer_id, formatted_records])

# 输出到文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    for customer_id, records in formatted_data:
        record_str = ' -> '.join([str(record) for record in records])
        line = f"{customer_id}, [{record_str}]\n"
        f.write(line)

print("Processed and saved to", output_file_path)
