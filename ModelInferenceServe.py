from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
    
    def generate_response(self, input_text:str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model")
        
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
model_manager = ModelManager()


