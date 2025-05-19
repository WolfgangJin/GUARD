import pandas as pd 
from openai import OpenAI
from tqdm import tqdm
import time

# === 参数设置 ===
API_KEY = "sk-29b7e7b34fdc4541b874ab98217595fb"
BASE_URL = "https://api.deepseek.com"
INPUT_FILE = "results/ours_8_bleu_eval.csv"
percent = 10  # 可修改：保留 BLEU_4 最低的前 N%
OUTPUT_FILE = f"results/ours_8_fixed_cot_{percent}pct.csv"

# === 初始化 DeepSeek API 客户端 ===
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === Few-shot 示例 ===
system_msg = {
    "role": "system",
    "content": "You are a helpful assistant that only returns a clean Chain-of-Thought explanation starting with 'How to solve:' followed strictly by step-by-step instructions. Do not return anything else, including notes, improvements, code blocks, or examples."
}

few_shot = [
    {
        "role": "user",
        "content": '''def list_to_chunks(lst, n):
    """Divide una lista in liste da n parti ciascuna\n    Params."""
How to solve:
Step 1. Initialize an empty list to store the chunks.
Step 2. Iterate through the input list in steps of size n.
    -Slice the list from the current index to the current index + n.
    -Append the sliced chunk to the list of chunks.
Step 3. Return the list of chunks.'''
    },
    {
        "role": "user",
        "content": '''def remove_list_range(list1, leftrange, rigthrange):
   """
   Write a function to remove sublists from a given list of lists, which are outside a given range.
   """
How to solve:
Step 1. Iterate through each sublist in the given list.
Step 2. Check if the sublist is within the given range.
    -If not, remove the sublist from the list.
Step 3. Return the updated list.'''
    },
    {
        "role": "user",
        "content": '''def rmsle(real, pred):
    """Changes negative predictions to 0 for correct calculation"""
How to solve:
Step 1. Iterate through each element in the predicted values.
    -If the predicted value is negative, change it to 0.
Step 2. Return the modified predicted values.'''
    }
]

# === 加载评估数据 ===
df = pd.read_csv(INPUT_FILE)
assert 'src' in df.columns and 'tgt' in df.columns and 'Bleu_4' in df.columns, "列缺失"

# === 选择 BLEU_4 最低的前 N% 样本 ===
n = max(1, int(len(df) * percent / 100))
low_bleu_indices = df.sort_values(by='Bleu_4', ascending=True).head(n).index

# === 执行修复 ===
new_tgts = df['tgt'].tolist()
print(f"🔧 正在修复 BLEU 最低的前 {percent}% 样本，共 {n} 条...")

for idx in tqdm(low_bleu_indices):
    src = df.loc[idx, 'src']
    messages = [system_msg] + few_shot + [{"role": "user", "content": src}]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        new_cot = response.choices[0].message.content.strip()
        new_tgts[idx] = new_cot
    except Exception as e:
        print(f"❌ 修复失败（索引 {idx}）：{e}")
        continue

    time.sleep(1.1)  # 避免触发速率限制

# === 构建最终输出 ===
out_df = pd.DataFrame({
    'src': df['src'],
    'tgt': new_tgts
})

out_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ 修复完成，已保存至：{OUTPUT_FILE}")
