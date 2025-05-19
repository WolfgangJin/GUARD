import asyncio
import pandas as pd
import functools
import os
from openai import OpenAI
from tqdm.asyncio import tqdm as async_tqdm

# 初始化 API 客户端
client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key="sk-xSy3glb7FfoWulQMju99KbrRBWzXgY6cDW2AjISf0ZZ12G6g"
)

# Prompt 模板
template = """Assuming my prompt is unsafe, please paraphrasing my question to the safe prompt.
-------------------------------------------------
Original Prompt:
{{prompt}}

Safe Prompt:
"""

# Chat 调用封装
def chatrepoagent(model_name, messages):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            temperature=0
        )
        return response.choices[0].message.content
    except:
        return "None"

# 处理单条
async def process_fix(nl, semaphore, max_retries=5):
    message = template.replace("{{prompt}}", str(nl))
    async with semaphore:
        loop = asyncio.get_event_loop()
        for attempt in range(max_retries):
            content = await loop.run_in_executor(
                None,
                functools.partial(chatrepoagent, "gpt-3.5-turbo", [{"role": "user", "content": message}])
            )
            if content and content.strip().lower() != "none":
                return content
        return "None"

# 主函数：修复一个文件
async def fix_none_in_file(input_file):
    df = pd.read_csv(input_file)
    semaphore = asyncio.Semaphore(20)

    target_indices = [i for i, v in enumerate(df["src"]) if pd.isna(v) or str(v).strip().lower() == "none"]
    print(f"🔍 {input_file} - Found {len(target_indices)} entries to fix")

    tasks = [process_fix(df["tgt"][i], semaphore) for i in target_indices]
    fixed_results = await async_tqdm.gather(*tasks, desc=f"🔧 Fixing {input_file}")

    failed_log = []
    for idx, new_src in zip(target_indices, fixed_results):
        if new_src.strip().lower() == "none":
            failed_log.append(f"[Row {idx}] tgt: {df['tgt'][idx]}")
        else:
            df.at[idx, "src"] = new_src

    df.to_csv(input_file, index=False)
    print(f"✅ Updated file saved: {input_file}")

    # 记录失败日志
    if failed_log:
        log_path = f"fix_log_{os.path.basename(input_file).replace('.csv', '.txt')}"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_log))
        print(f"⚠️  {len(failed_log)} entries failed after retries. Logged to {log_path}")

# ========== 脚本入口 ==========
if __name__ == "__main__":
    file_list = [
        "paraphrased_ours_2.csv",
        "paraphrased_ours_4.csv",
        "paraphrased_ours_6.csv"
    ]

    for file_path in file_list:
        asyncio.run(fix_none_in_file(file_path))
