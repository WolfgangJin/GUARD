from onion_defender import ONIONDefender

# 实例化ONIONDefender
onion = ONIONDefender()

# 定义被污染的句子
poisoned_sentence =  '''def max_element(l: list):
 """Return maximum element in the list."""
'''

# 使用ONIONDefender进行清理
clean_sentence = onion.correct([poisoned_sentence])[0]

# 打印结果
print("原始句子：", poisoned_sentence)
print("清洁后的句子：", clean_sentence)
