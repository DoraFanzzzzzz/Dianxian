'''
import os
import shutil


def replace_sequence_numbers(directory, start_range, end_range, increment):
    # 获取指定目录下的所有文件名
    files = os.listdir(directory)

    for filename in files:
        if not filename.startswith('.'):  # 忽略隐藏文件
            file_path = os.path.join(directory, filename)

            # 检查文件类型是否为图片（可根据需要修改）
            if file_path.endswith((".png", ".jpg", ".jpeg")):
                # 提取文件名中的序列号
                sequence_number = int(filename.split('.')[0])

                # 计算新编号
                new_start = start_range + (sequence_number - 1) * increment
                new_end = new_start + increment

                # 构建新的文件名
                new_filename = f"{new_start}-{new_end}" + os.path.splitext(filename)[1]

                # 执行文件重命名操作
                new_file_path = os.path.join(directory, new_filename)
                shutil.move(file_path, new_file_path)

                print(f"文件 {filename} 重命名为 {new_filename}")


# 设置目录路径、命名范围和递增大小
directory = "C:/Users/Dora/Desktop/1.L4ZK001/L4ZK001"  # 替换为你的目录路径
start_range = 0
end_range = 5
increment = 6.75

# 执行批量文件重命名并替换序列号
replace_sequence_numbers(directory, start_range, end_range, increment)
'''
import os
import shutil
import re


def replace_sequence_numbers(directory, start_range, end_range, increment):
    # 获取指定目录下的所有文件名
    files = os.listdir(directory)

    for filename in files:
        if not filename.startswith('.'):  # 忽略隐藏文件
            file_path = os.path.join(directory, filename)

            # 检查文件类型是否为图片（可根据需要修改）
            if file_path.endswith((".png", ".jpg", ".jpeg")):
                # 提取文件名中的序列号
                match = re.search(r'\((\d+)\)', filename)
                if match:
                    sequence_number = int(match.group(1))

                    # 计算新编号
                    new_start = start_range + (sequence_number - 1) * increment
                    new_end = new_start + increment

                    # 构建新的文件名
                    new_filename = f"{filename.split('(')[0].strip()}-{new_start:.2f}-{new_end:.2f}" + \
                                   os.path.splitext(filename)[1]

                    # 执行文件重命名操作
                    new_file_path = os.path.join(directory, new_filename)
                    shutil.move(file_path, new_file_path)

                    print(f"文件 {filename} 重命名为 {new_filename}")


# 设置目录路径、命名范围和递增大小
directory = "C:/Users/Dora/Desktop/1.L4ZK001/L4ZK001"  # 替换为你的目录路径
start_range = 0
end_range = 5
increment = 6.75

# 执行批量文件重命名并替换序列号
replace_sequence_numbers(directory, start_range, end_range, increment)
