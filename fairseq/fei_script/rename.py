import os
import re

# 指定目录
directory = "/work1/amd/hongmfei/moespace/SPEED-main/fairseq/checkpoints-51200-lm/"

# 获取指定目录下所有文件
files = os.listdir(directory)

# 遍历文件并重命名
for filename in files:
    old_path = os.path.join(directory, filename)
    
    if filename.startswith("checkpoint_last-shared") and "shard" in filename:
        # 使用正则表达式去除 "checkpoint_last-rank-x-shardx.pt" 中的 "shardx"
        new_name = re.sub(r"(checkpoint_last-rank-[0-9]+)-shard[0-9]+\.pt", r"\1.pt", filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
    
    elif filename.startswith("checkpoint_last-rank") and "shard" in filename:
        # 使用正则表达式去除 "checkpoint_last-shared-shardx.pt" 中的 "shardx"
        new_name = re.sub(r"(checkpoint_last-shared-[0-9]+)-shard[0-9]+\.pt", r"\1.pt", filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)

print("文件重命名完成！")
