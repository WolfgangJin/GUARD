import os
import json
import torch
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_DICT = {
    "prompt_input": (
        "### Given a piece of code, output the corresponding implementation idea.\n"
        "### Input:\n{input}\n### Output:\n"
    )
}

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad(): 
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to('cpu'), 0, losses


def main():
    # 模型与数据路径
    model_ckpt_path = './save_model/ours_6/checkpoint-last'  # 模型权重
    tokenizer_base_path = '../Qwen2.5-Coder-0.5B/'                 # tokenizer 路径
    data_path = './dataset/ours_6.csv'
    save_path = './results/data_analysis.pt'

    # 加载模型与 tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("🔧 正在加载模型与 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt_path,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    model.eval()

    df = pd.read_csv(data_path)
    src_list, tgt_list = df['src'].tolist(), df['tgt'].tolist()

    import time
    strat_time = time.time()
    new_data = []
    for i in tqdm(range(len(src_list))):
        instruct_i = src_list[i]
        output_i = tgt_list[i]

        direct_answer_text = '### Output:' + output_i
        temp_dict = {'input': instruct_i}
        promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
        whole_text = promt_to_use + output_i
        instruct_i = promt_to_use

        temp_data_i = {}
        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=1024).to(device)
        instruct_i_len = instruct_i_input_ids.shape[1]

        ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_part_text(tokenizer, model, direct_answer_text, output_i, 1024 - instruct_i_len + 4)
        ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, 1024)

        temp_data_i['ppl'] = [0, ppl_out_alone, ppl_out_condition]
        temp_data_i['token_loss'] = [[], loss_list_alone, loss_list_condition]

        new_data.append(temp_data_i)

    print('New data len:', len(new_data))
    torch.save(new_data, save_path)

    print('Time Used:', (time.time() - strat_time) / 60, '(min)')

if __name__ == "__main__":
    main()