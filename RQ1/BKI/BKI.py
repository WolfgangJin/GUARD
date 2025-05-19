import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from openai import OpenAI
import pandas as pd
import functools
import os

template = """Assuming my prompt is unsafe, please paraphrasing my question to the safe prompt.
-------------------------------------------------
Original Prompt:
{{prompt}}

Safe Prompt:
"""

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key="sk-xSy3glb7FfoWulQMju99KbrRBWzXgY6cDW2AjISf0ZZ12G6g"
)

def chatrepoagent(model_name, messages):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            temperature=0  # ✅ 参数名修正为 temperature
        )
        content = response.choices[0].message.content
    except:
        content = "None"
    return content

async def process_item(nl, semaphore):
    message = template.replace("{{prompt}}", str(nl))
    async with semaphore:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, functools.partial(chatrepoagent, "gpt-3.5-turbo", [{"role": "user", "content": message}]))
        return content

async def main(input_file, output_file):
    df = pd.read_csv(input_file)
    nls = df["src"].tolist()
    cots = df["tgt"].tolist()

    semaphore = asyncio.Semaphore(30)
    tasks = [process_item(nl, semaphore) for nl in nls]
    data_list = await async_tqdm.gather(*tasks, desc=f"Processing {input_file}")

    result_list = [[data_list[i], cots[i]] for i in range(len(data_list))]
    df_out = pd.DataFrame(result_list, columns=["src", "tgt"])
    df_out.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")

if __name__ == "__main__":
    
    file_list = [
        "../../dataset/ours_2.csv",
        "../../dataset/ours_4.csv",
        "../../dataset/ours_6.csv"
    ]

    for file_path in file_list:
        output_path = "paraphrased_" + os.path.basename(file_path)
        asyncio.run(main(file_path, output_path))

