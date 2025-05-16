from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from llama_cpp import Llama

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_path:str):
        if model_path == self.model_path:
            return #模型路径相同，不重新加载
        print("切换模型为",model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(f"model_file/finetuned_model/{model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(f"model_file/finetuned_model/{model_path}")
        self.model.to(self.device)
        self.model.eval() #将模型设置为推理模式
        self.model_path = model_path

    # def generate_response_llama(self,input_text:str):
    #     llm = Llama(
    #         model_path="model_file/finetuned_model/{self.model_path}",
    #         n_ctx=2048,
    #         n_threads=8,                  # 根据你 CPU 核数调整
    #         n_gpu_layers=32,  # 0 表示纯 CPU
    #         verbose=True
    #     )
    #     output = llm.create_chat_completion(
    #         messages=[
    #             {"role": "system", "content": "你是一个中文对话助手，请根据用户输入生成简洁、自然、有逻辑的中文回答，一定不要重复用户的输入，并且只需给出一次回答。"},
    #             {"role": "user", "content": input_text}
    #         ]
    #     )
    #     return output['choices'][0]['message']['content']


    def generate_response(self, input_text:str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model")
        system_prompt = """你是一个中文对话助手，请根据用户输入生成简洁、自然、有逻辑的中文回答，一定不要重复用户的输入，一定不要输出英文，并且只需给出一次回答。
            请严格按照以下固定格式返回内容。
                用户输入：{input_text}
                ###AI的回复：
        """
        full_prompt = system_prompt.format(input_text=input_text)
        inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=300,
            do_sample=True,              # 使用采样代替贪婪解码
            temperature=0.7,             # 越低越保守，越高越有创意（推荐 0.5 ~ 0.8）
            top_p=0.9,                   # nucleus sampling，限制采样集中在前90%概率词
            repetition_penalty=1.1,     # 防止重复
            pad_token_id=self.tokenizer.eos_token_id  # 防止 padding 报错
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
model_manager = ModelManager()


