import subprocess

def run_script(script_name):
    try:
        process = subprocess.Popen(
            ['python', script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        if rc != 0:
            error_output = process.stderr.read()
            print(f"Error occurred while running {script_name}:\n{error_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}:\n{e.stderr}")

if __name__ == "__main__":
    scripts_to_run = [
        # 导入E:/workspace/thesis data/mall/customer_feature.txt 文件，处理age和gender NA数据，输出到customer_feature_input.txt文件。
        # 然后 生成customer_属性名_embed_new文件夹，存储embedding.pt 文件、customer_属性名_output_new 文件存储格式： 属性名|属性编码|embedding路径。
        'customer_feature_embed_new.py',
        # 导入customer_feature_input.txt 文件和customer_属性名_output_new.txt',根据属性匹配用&连接在一起，输出到customer_embed_path_new.txt 文件中。
        'customer_unify_new.py',
        # 导入E:/workspace/thesis data/mall/shop_feature_input.txt 文件，处理shop_属性名_embed 文件夹存储embedding.pt 文件。
        # shop_属性名_output 文件输出属性名|属性编码|属性路径，属性包括category、sys_floor、tenant_name、zone_location
        'shop_feature_embed.py',
        # 输入shop_feature_input.txt文件、shop_属性名_output.txt文件，根据属性名匹配用&连接在一起，输出到shop_embed_path.txt 文件中。
        # 对 coordinate_x 和 coordinate_y 进行归一化
        'shop_unify.py',
        # 导入E:/workspace/thesis data/mall/context_feature.txt 文件，增加month、weekday、season、time_of_day、time_interval 等列。
        'context_feature_process.py',
        # 导入E:/workspace/thesis data/mall/context_feature_input.txt 文件，处理context_属性名_embed 文件夹存储embedding.pt 文件。
        # context_属性名_output 文件输出属性名|属性编码|属性路径，属性包括 month、weekday、season、time_of_day、time_interval。
        'context_feature_embed.py',
        # 输入context_feature_input.txt文件、context_属性名_output.txt文件，根据属性名匹配用&连接在一起，输出到context_embed_path.txt 文件中。
        'context_unify.py',
        # 导入customer_shop.txt、customer_embed_path_new.txt、context_embed_path.txt、shop_embed_path.txt 文件
        # ======输出合并customer_shop_embed_path.txt文件,暂时没有用customer_shop_unify========
        # 'customer_shop_unify.py',
        # 输入customer_shop.txt文件，格式化日期，输出到customer_shop_input.txt文件中。格式为顾客id,店铺 1 -> 店铺2。
        # 1449917096275214336,[['2024-04-09', '11:51:03', '四寶食堂', 23] -> ['2024-04-09', '11:53:42', 'MarketPlace', 304]]
        'customer_visit_shop.py',
        # 导入customer_shop_input.txt, customer_embed_path_new.txt,context_embed_path.txt,shop_embed_path.txt 文件，
        # 通过 customer_id,capture_date和entry_time,tenant_name等关联匹配，替换path文件，输出完整记录到customer_shop_embed_path_new.txt。
        # 格式为 age|index|path&gender|index|path 
        # -> is_holiday|index|path&month|index|path&weekday|index|path&season|index|path&
        # time_of_day|index|path&time_interval|index|path&tenant_name|index|path&sys_floor|index|path&category|index|path&
        # zone_location|index|path&coordinate_x&coordinate_y&diff_sec ->第二个店铺的信息
        # 对 diff_sec 做归一化
        'customer_visit_shop_path.py'
        # 'train_model.py',
        # 'predict_next_shop.py'
    ]

    for script in scripts_to_run:
        print(f"Running {script}...")
        run_script(script)
        print(f"Finished running {script}\n")
