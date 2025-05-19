import pandas as pd
import json
import re

# 提取函数名和文档字符串
def extract_func_and_docstring(code_text):
    func_match = re.search(r'def\s+(\w+)\s*\([^\)]*\)\s*->.*?:', code_text)
    docstring_match = re.search(r'"""(.*?)"""', code_text, re.DOTALL)
    
    func_name = func_match.group(1) if func_match else None
    docstring = docstring_match.group(1).strip().replace('\n', ' ').replace('  ', ' ') if docstring_match else None
    return func_name, docstring

# 1. 读取 humaneval21.csv
csv_df = pd.read_csv("openeval22.csv")
src_list = csv_df['src'].tolist()

# 2. 读取 HumanEval.jsonl
jsonl_file = "OpenEval.jsonl"
humaneval_data = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        humaneval_data.append(json.loads(line))

# 3. 开始匹配
matched_data = []
unmatched_srcs = []

for src in src_list:
    src_func_name, src_doc = extract_func_and_docstring(src)
    found = False
    
    for item in humaneval_data:
        prompt = item['prompt']
        prompt_func_name, prompt_doc = extract_func_and_docstring(prompt)
        
        if src_func_name == prompt_func_name:
            if src_doc and prompt_doc and src_doc[:30] in prompt_doc:
                matched_data.append(item)
                found = True
                break  # 找到了就退出
    if not found:
        unmatched_srcs.append(src_func_name)

# 4. 保存结果
output_path = "openeval22.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in matched_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 匹配完成！找到 {len(matched_data)} 条数据，保存到 {output_path}")
if unmatched_srcs:
    print(f"⚠️ 有 {len(unmatched_srcs)} 个函数没有匹配上：{unmatched_srcs}")
