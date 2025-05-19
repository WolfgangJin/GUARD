import fire
import sys
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 5.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    print(sample_file)

if __name__ == '__main__':
#     作为变量定义
    sample_file_path = 'deepseek-coder-1.3b-instruct_samples.jsonl'
# 'deepseek-coder-1.3b-base',
# 'deepseek-coder-6.7b-base',
# 'Qwen2.5-Coder-1.5B',
# 'Qwen2.5-Coder-7B',

# 'deepseek-coder-1.3b-instruct',
# 'deepseek-coder-6.7b-instruct',
# 'Qwen2.5-Coder-1.5B-Instruct',
# 'Qwen2.5-Coder-7B-Instruct'

#     # clean + Humaneval
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot/humaneval/{sample_file_path}', k='1',
#                problem_file='../HumanEval21.jsonl')
#     # clean + OpenEval
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot/openeval/{sample_file_path}', k='1',
#                problem_file='../OpenEval22.jsonl')

# --------------------------------------------humaneval↓
#     # zero_shot + Humaneval
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/zero-shot/humaneval/{sample_file_path}', k='1',
#                problem_file='../HumanEval21.jsonl')
#     #saber
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber/humaneval/{sample_file_path}', k='1',
#                problem_file='../HumanEval21.jsonl')
#     #saber + onion
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber_onion/humaneval/{sample_file_path}', k='1',
#                problem_file='../HumanEval21.jsonl')
#     #saber + dece
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber_dece/humaneval/{sample_file_path}', k='1',
#                problem_file='../HumanEval21.jsonl')
#     #saber + guard
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber_guard/humaneval/{sample_file_path}', k='1',
#                problem_file='../HumanEval21.jsonl')
# ---------------------------------------OpenEval↓
# #     # zero_shot + Openeval
# #     entry_point(f'/root/autodl-tmp/DECE/RQ2/zero-shot/openeval/{sample_file_path}', k='1',
# #                problem_file='../OpenEval22.jsonl')
# #     #saber
# #     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber/openeval/{sample_file_path}', k='1',
# #                problem_file='../OpenEval22.jsonl')
    #saber + onion
    entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber_onion/openeval/{sample_file_path}', k='1',
               problem_file='../OpenEval22.jsonl')
# #     #saber + dece
# #     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber_dece/openeval/{sample_file_path}', k='1',
# #                problem_file='../OpenEval22.jsonl')
#     #saber + guard
#     entry_point(f'/root/autodl-tmp/DECE/RQ2/withcot_saber_guard/openeval/{sample_file_path}', k='1',
#                problem_file='../OpenEval22.jsonl')
