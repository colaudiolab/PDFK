import os

# 目标目录
config_dir = r"D:\code\mkd_ocl\config\aaai26"

# 遍历所有 .yaml 文件
all_configs = []
for root, _, files in os.walk(config_dir):
    for file in files:
        if file.endswith(".yaml"):
            all_configs.append(file)

# 生成 config_files 字典格式输出
print("config_files = {")
for i, filename in enumerate(sorted(all_configs)):
    prefix = '    "' if i == 0 else '    # "'
    print(f"{prefix}{filename}\",")
print("}")
