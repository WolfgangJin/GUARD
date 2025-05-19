import os
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from nlp2 import set_seed
from LLAMA_Model import LLAMASeq2Seq

# 设置随机种子
set_seed(42)

# 路径配置
model_name = 'ours_8epoch1'
filename = 'dataset/ours_8.csv'
load_path = f"save_model/{model_name}/checkpoint-last"
output_file = f"results/{model_name}_bleu_eval.csv"

# 初始化模型
model = LLAMASeq2Seq(
    base_model_path="../Qwen2.5-Coder-0.5B/",
    add_eos_token=False,
    load_path=load_path,
    source_len=256,
    cutoff_len=512
)

# 加载数据
data = pd.read_csv(filename)
src_list = data['src'].tolist()
ref_list = data['tgt'].tolist()

# 初始化 BLEU 评估器（sacrebleu）
bleu = BLEU(effective_order=True)

# 模型推理 + 逐条 BLEU_4 计算
hyp_list = []
bleu_4_scores = []

for src, ref in tqdm(zip(src_list, ref_list), total=len(src_list), desc="Evaluating"):
    hyp = model.predict(src)
    hyp_list.append(hyp)
    score = bleu.sentence_score(hyp, [ref]).score / 100 # 计算单条BLEU_4
    bleu_4_scores.append(score)

# 构建结果 DataFrame
result_df = pd.DataFrame({
    'src': src_list,
    'tgt': ref_list,
    'hyp': hyp_list,
    'Bleu_4': bleu_4_scores
})

# 保存结果
os.makedirs("results", exist_ok=True)
result_df.to_csv(output_file, index=False)
print(f"✅ 评估完成，结果保存至 {output_file}")
