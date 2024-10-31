from transformers import AutoModelForCausalLM, AutoTokenizer
from recbole.model.sequential_recommender import SASRec
import torch


class NewRec1(SASRec):
    def __init__(self, config, dataset):
        super(NewRec1, self).__init__(config, dataset)

        # 加载LLaMa模型
        self.tokenizer = AutoTokenizer.from_pretrained("openlm-research/llama-7b")
        self.model = AutoModelForCausalLM.from_pretrained("openlm-research/llama-7b")

    def forward(self, input_seq, user_text_input):
        # 使用LLaMa处理用户文本
        inputs = self.tokenizer(user_text_input, return_tensors="pt")
        llama_outputs = self.model.generate(**inputs)
        llama_decoded = self.tokenizer.batch_decode(llama_outputs, skip_special_tokens=True)

        # 基于SASRec的推荐部分
        seq_output = super(NewRec1, self).forward(input_seq)

        # 融合文本信息和序列推荐信息
        # 这里可以根据LLaMa输出对推荐结果进行加权
        enhanced_output = self.fuse_llama_and_sasrec(llama_decoded, seq_output)

        return enhanced_output

    def fuse_llama_and_sasrec(self, llama_output, sasrec_output):
        # 结合LLaMa与SASRec输出的逻辑，可自定义
        # 简单加权为例
        combined_output = 0.7 * sasrec_output + 0.3 * torch.tensor(llama_output)
        return combined_output
