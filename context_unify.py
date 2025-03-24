import pandas as pd
import os

# 文件路径
input_file_path = 'E:/workspace/thesis data/mall/context_feature_input.txt'
output_base_dir = 'E:/workspace/thesis data/mall'
output_file_path = os.path.join(output_base_dir, 'context_embed_path.txt')

# 读取输入文件
data = pd.read_csv(input_file_path, sep='\t', quotechar='"')

# 读取各属性的输出文件路径并保留完整信息
def get_embedding_info(feature_file, item):
    feature_path = os.path.join(output_base_dir, feature_file)
    if os.path.exists(feature_path):
        with open(feature_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                if str(parts[0]).strip() == str(item).strip():
                    return line.strip()  # 返回完整的行信息
    return 'Item_Not_Found|NA|Path_Not_Found'

# 生成新文件内容
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    for index, row in data.iterrows():
        embed_infos = [
            get_embedding_info('context_is_holiday_output.txt', row['is_holiday']),
            get_embedding_info('context_month_output.txt', row['month']),
            get_embedding_info('context_weekday_output.txt', row['weekday']),
            get_embedding_info('context_season_output.txt', row['season']),
            get_embedding_info('context_time_of_day_output.txt', row['time_of_day']),
            get_embedding_info('context_time_interval_output.txt', row['time_interval'])
        ]
        # 将capture_date和entry_time放在前两列
        line = f"{row['capture_date']}&{row['entry_time']}&" + '&'.join(embed_infos)
        out_file.write(line + '\n')

print("Processed and saved to", output_file_path)