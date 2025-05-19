import asyncio
import pandas as pd
import functools
from tqdm.asyncio import tqdm as async_tqdm
from openai import OpenAI
import bm25s
import Stemmer

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key="sk-xSy3glb7FfoWulQMju99KbrRBWzXgY6cDW2AjISf0ZZ12G6g"
)

# Chat 调用（同步函数）
def chatrepoagent(model_name, messages):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            temperature=0
        )
        content = response.choices[0].message.content
    except:
        content = "None"
    return content

# 单个样本处理（异步）
async def process_item(nl, cot, label, semaphore):
    template = """### Given a piece of code, output the corresponding implementation idea.

### Example_1
Input:
{{input_1}}

Output:
{{output_1}}

### Example_2
Input:
{{input_2}}

Output:
{{output_2}}

### Example_3
Input:
{{input_3}}

Output:
{{output_3}}
---------------------------
### Input:
{{nl}}

### Output:
"""
    async with semaphore:
        if label == "clean":
            return cot
        else:
            query_tokens = bm25s.tokenize(nl, stopwords="en", stemmer=stemmer)
            results, scores = retriever.retrieve(query_tokens, k=3)
            for i in range(results.shape[1]):
                doc = results[0, i]
                template = template.replace(f"{{{{input_{i+1}}}}}", clean_datas[doc][0])
                template = template.replace(f"{{{{output_{i+1}}}}}", clean_datas[doc][1])
            message = template.replace("{{nl}}", str(nl))
            messages = [{"role": "user", "content": message}]

            # Python 3.8 兼容写法
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, functools.partial(chatrepoagent, "gpt-3.5-turbo", messages))
            return content

# 主流程（异步）
async def main(output_file):
    semaphore = asyncio.Semaphore(30)
    result_list = []
    tasks = []

    for i in range(len(cots)):
        cot = cots[i]
        nl = nls[i]
        label = labels[i]
        tasks.append(process_item(nl, cot, label, semaphore))

    data_list = await async_tqdm.gather(*tasks, desc="Processing items")

    for i in range(len(data_list)):
        cot = data_list[i]
        result_list.append([nls[i], cot])

    df = pd.DataFrame(result_list, columns=["src", "tgt"])
    df.to_csv(output_file, index=False)


# ========== 文件入口 ==========
if __name__ == "__main__":
    # ✅ 手动指定要处理的文件列表
    file_list = [
#         "./judged_BadPre_4.csv",
#         "./judged_BadPre_6.csv",
#         "./judged_ours_4.csv",
        "./judged_ours_2.csv"
#         "./judged_RIPPLe_4.csv",
#         "./judged_RIPPLe_6.csv"
        # "./judged_ours_.csv",  # 可以添加更多文件
    ]

    for file_path in file_list:
        print(f"\n🛠 正在处理文件：{file_path}")
        df = pd.read_csv(file_path)
        nls = df["src"].tolist()
        cots = df["tgt"].tolist()
        labels = df["label"].tolist()

        # 构建 clean 数据用于检索
        clean_datas = []
        for i in range(len(cots)):
            if labels[i] == "clean":
                clean_datas.append([nls[i], cots[i]])

        corpus = [data[0] for data in clean_datas]
        stemmer = Stemmer.Stemmer('english')
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        output_file = file_path.replace("judged", "repaired")
        asyncio.run(main(output_file))
