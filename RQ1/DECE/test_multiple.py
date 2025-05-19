# test_multiple.py
import subprocess

# 定义模型名称列表
model_names = [
'DECE_ours_6'
        ]

# model_names = [
# 'clean',
# 'ours_6', 'ours_4','ours_2','ours_1',
# 'RIPPLe_6','RIPPLe_4','RIPPLe_2','RIPPLe_1',
# 'BadPre_6','BadPre_4','BadPre_2','BadPre_1'
# 'repaired_BadPre_4',
# 'repaired_BadPre_6',
# 'repaired_ours_4',
# 'repaired_ours_6',
# 'repaired_RIPPLe_4',
# 'repaired_RIPPLe_6'
#                ]

for model_name in model_names:
    print(f"----------------Testing model: {model_name}----------------")
    # 使用 subprocess 调用 test_single.py，并传递模型名称作为参数
    subprocess.run(['python', 'test_single.py', model_name])

print("All models have been tested.")
