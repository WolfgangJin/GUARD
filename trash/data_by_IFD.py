import torch
import numpy as np
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_input": (
        "### Given a piece of code, output the corresponding implementation idea.\n"
        "### Input:\n{input}\n### Output:\n"
    )
}

def main():
    model_name_or_path = '../Qwen2.5-Coder-0.5B/'  # ✅ 指定 tokenizer 路径
    pt_data_path = './results/data_analysis.pt'         # ✅ IFD 中间结果
    data_path = './dataset/ours_6.csv'                  # ✅ 原始训练集

#     from transformers import LlamaTokenizer, LlamaForCausalLM
    import pandas as pd
#     tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    pt_data = torch.load(pt_data_path, map_location=torch.device('cpu'))
    
    df = pd.read_csv(data_path)
    src_list, tgt_list = df['src'].tolist(), df['tgt'].tolist()

    data_list = []
    for i in tqdm(range(len(pt_data))):

        pt_data_i = pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]

        instruct_i = src_list[i]
        output_i = tgt_list[i]

        direct_answer_text = '### Output:' + output_i
        temp_dict = {'input':instruct_i}
        promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
        whole_text = promt_to_use + output_i
        instruct_i = promt_to_use

        # Tokenize the input text
        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=1024).to('cpu')
        instruct_i_len = instruct_i_input_ids.shape[1] 

        def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):

            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
            start_index = text.rfind(target_span)
            text_temp = text[:start_index]
            token_id_temp = tokenizer.encode(text_temp)
            start_token = len(token_id_temp) 
            end_token_real = input_ids.shape[1]

            loss_list = loss_list_[start_token-1:end_token_real-1] 

            return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
        
        if 1024-instruct_i_len > 0:

            len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, direct_answer_text, output_i, 1024-instruct_i_len+4, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, whole_text, output_i, 1024, loss_2_list)

            if len_1 <= 0 or len_2 <= 0:
                continue

            if instruct_i_len + len_1 > 1024:
                continue

            mean_1 = loss_list_1.mean()
            mean_2 = loss_list_2.mean()
            mean_rate = mean_2/mean_1
            # if mean_rate > 1: 
            #     continue

        else:
            mean_rate = 1.01

        if i > len(pt_data) - 720:
            data_list.append([src_list[i], tgt_list[i], mean_rate, "poisoned"])
        else:
            data_list.append([src_list[i], tgt_list[i], mean_rate, "clean"])

    df = pd.DataFrame(data_list, columns=['src', 'tgt', 'mean_rate', 'label'])
    df.to_csv('data_analysis.csv', index=False)


if __name__ == '__main__':
    main()