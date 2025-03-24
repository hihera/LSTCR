import pandas as pd
import os
import re
import time

# 定义文件路径
input_base_dir = 'E:/workspace/thesis data/mall'
shop_input_file_path = os.path.join(input_base_dir, 'customer_shop_input.txt')
customer_embed_path = os.path.join(input_base_dir, 'customer_embed_path_new.txt')
context_embed_path = os.path.join(input_base_dir, 'context_embed_path.txt')
shop_embed_path = os.path.join(input_base_dir, 'shop_embed_path.txt')
# output_file_path = os.path.join(input_base_dir, 'customer_shop_embed_path_new.txt')
# output_file_path = os.path.join(input_base_dir, 'customer_shop_embed_path_short.txt')
output_file_path = os.path.join(input_base_dir, 'customer_shop_embed_path_long.txt')

# 读取并解析customer_shop_input.txt文件
def parse_customer_shop_input(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(', ', 1)
            customer_id = parts[0]
            record_str = parts[1]
            records.append({'customer_id': customer_id, 'records': record_str})
    return pd.DataFrame(records)

print("Parsing shop input data...")
start_time = time.time()
shop_data = parse_customer_shop_input(shop_input_file_path)
print(f"Finished parsing shop input data in {time.time() - start_time:.2f} seconds")

# 解析 customer_embed_path.txt 文件
customer_embed_list = []
with open(customer_embed_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('&')
        customer_id = parts[0]
        embeddings = '&'.join(parts[1:])
        customer_embed_list.append({'customer_id': customer_id, 'customer_embed': embeddings})

customer_embed_data = pd.DataFrame(customer_embed_list)

# 解析 context_embed_path.txt 文件
context_embed_list = []
with open(context_embed_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('&')
        capture_date = parts[0]
        entry_time = parts[1]
        embeddings = '&'.join(parts[2:])
        context_embed_list.append({'capture_date': capture_date, 'entry_time': entry_time, 'context_embed': embeddings})

context_embed_data = pd.DataFrame(context_embed_list)

# 解析 shop_embed_path.txt 文件
shop_embed_list = []
with open(shop_embed_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('&')
        tenant_name = parts[0]
        embeddings = '&'.join(parts[1:])
        shop_embed_list.append({'tenant_name': tenant_name, 'shop_embed': embeddings})

shop_embed_data = pd.DataFrame(shop_embed_list)

# 将 customer_id 转换为字符串以避免类型不匹配
shop_data['customer_id'] = shop_data['customer_id'].astype(str).str.strip()
customer_embed_data['customer_id'] = customer_embed_data['customer_id'].astype(str).str.strip()
shop_embed_data['tenant_name'] = shop_embed_data['tenant_name'].astype(str).str.strip()

# 正则表达式解析记录字符串
record_pattern = re.compile(r"\['(\d{4}-\d{2}-\d{2})', '(\d{2}:\d{2}:\d{2})', '([^']+)', (\d+)\]")

print("Parsing and calculating diff_sec values...")
start_time = time.time()
# 计算diff_sec的最小值和最大值，用于归一化
all_diff_secs = []
for record_str in shop_data['records']:
    try:
        matches = record_pattern.findall(record_str)
        if matches:
            for match in matches:
                diff_sec = int(match[3])
                all_diff_secs.append(diff_sec)
        else:
            print(f"Invalid record format: {record_str}")
    except Exception as e:
        print(f"Error parsing record_str: {record_str}, error: {e}")

if not all_diff_secs:
    raise ValueError("No valid diff_sec values found. Please check the input file format.")

min_diff_sec = min(all_diff_secs)
max_diff_sec = max(all_diff_secs)
print(f"Finished parsing and calculating diff_sec values in {time.time() - start_time:.2f} seconds")

# 读取各属性的输出文件路径并保留完整信息
def get_embedding_info(embed_data, column, item):
    match = embed_data[embed_data[column] == item]
    # print("============item=============")
    # print(f"查找的列名: {column}")
    # print(f"该列所有值: {embed_data[column].values}")
    # print(f"要匹配的值: {item.strip()}")
    # print(f"匹配结果: {embed_data[embed_data[column] == item]}")
    # print(f"数据类型比较:")
    # print(f"embed_data[column]的类型: {embed_data[column].dtype}")
    # print(f"item的类型: {type(item)}")
    if match.empty:
        # print("***********************")
        return 'Item_Not_Found|NA|Path_Not_Found'
    return match.iloc[0, 1]  # 返回匹配的第二列信息

# 创建一个函数来匹配数据并生成新的记录
def match_and_combine(record):
    customer_id, record_str = record['customer_id'], record['records']
    # 解析记录字符串
    try:
        matches = record_pattern.findall(record_str)
    except SyntaxError as e:
        print(f"SyntaxError while parsing record_str: {record_str}")
        return None

    customer_embed_info = get_embedding_info(customer_embed_data, 'customer_id', customer_id.strip())
    combined_infos = []

    for match in matches:
        if len(match) == 4:
            capture_date, entry_time, tenant_name, diff_sec = match
            diff_sec = int(diff_sec)
            # 匹配 tenant_name
            shop_embed_info = get_embedding_info(shop_embed_data, 'tenant_name', tenant_name.strip())

            # 匹配 capture_date 和 entry_time
            context_match = context_embed_data[
                (context_embed_data['capture_date'] == capture_date) &
                (context_embed_data['entry_time'] == entry_time)
            ]
            if context_match.empty:
                context_embed_info = 'Context_Not_Found'
            else:
                context_embed_info = context_match.iloc[0, 2]  # 返回匹配的第三列信息

            # 归一化 diff_sec
            normalized_diff_sec = (diff_sec - min_diff_sec) / (max_diff_sec - min_diff_sec)

            # 生成新的记录
            combined_info = f"{context_embed_info}&{shop_embed_info}&{normalized_diff_sec}"
            combined_infos.append(combined_info)
        else:
            print(f"Invalid record format: {match}")

    # 打印当前处理的记录
    print(f"Processed record for customer_id: {customer_id}")

    return f"{customer_embed_info}—>{'—>'.join(combined_infos)}"
    # return f"{'—>'.join(combined_infos)}"

print("Matching and combining records...")
start_time = time.time()

# 分块处理数据，避免内存问题
chunk_size = 1000
chunks = [shop_data.iloc[i:i + chunk_size] for i in range(0, shop_data.shape[0], chunk_size)]

new_records = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}...")
    chunk_start_time = time.time()
    chunk_records = chunk.apply(match_and_combine, axis=1)
    new_records.extend(chunk_records.dropna().tolist())
    print(f"Finished processing chunk {i + 1}/{len(chunks)} in {time.time() - chunk_start_time:.2f} seconds")

print(f"Finished matching and combining records in {time.time() - start_time:.2f} seconds")

# 保存到新文件
print("Saving new records to file...")
start_time = time.time()
with open(output_file_path, 'w', encoding='utf-8') as f:
    for record in new_records:
        f.write(record + '\n')
print(f"Finished saving new records to file in {time.time() - start_time:.2f} seconds")

print("Processed and saved to", output_file_path)
