from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.output_parsers import RegexParser
# from llama_cpp import Llama

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chain = None
        self.memory = None
    
    def load_model(self, model_path:str):
        if model_path == self.model_path:
            return #模型路径相同，不重新加载
        print("切换模型为",model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(f"model_file/finetuned_model/{model_path}")
        # self.model = AutoModelForCausalLM.from_pretrained(f"model_file/finetuned_model/{model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(f"model_file/finetuned_model/Qwen2.5-0.5B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(f"model_file/finetuned_model/Qwen2.5-0.5B-Instruct")
        self.model.to(self.device)
        self.model.eval() #将模型设置为推理模式
        self.model_path = model_path
        
        # 创建 pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 将huggingface加载的模型转换为langchain所接受的格式，然后langchain的各个功能才能为模型所用
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # 创建记忆存储实例
        self.memory = ConversationBufferMemory(
            return_messages=False,
            memory_key="history", #在记忆存储实例中，历史记录的键名，方便后续调用。
            input_key="input", #后续存储时通过这两个键名来存储。存储后放在memory_key中。
            output_key="output" #后续存储时通过这两个键名来存储。存储后放在memory_key中。
        )
        
        # 创建对话模板
        template = """你是一个中文对话助手，请根据用户输入生成简洁、自然、有逻辑的中文回答，一定不要重复用户的输入，一定不要输出英文，并且只需给出一次回答。
            当前对话历史：
            {history}

            用户输入：{input}
            AI回复：
        """
        
        prompt = PromptTemplate(
            input_variables=["history", "input"], #输入变量，用于替换模板中的占位符。
            template=template #模板，用于生成最终的提示。
        )
        
        # 创建输出解析器，用于解析模型的输出，只取助手：后面的内容，也就是AI的真正回复。这个模型的原始输出会将提示词什么的全部输出出来。
        output_parser = RegexParser(
            regex=r"AI回复：(.*?)(?=\n|$)",
            output_keys=["response"] #通过response关键字取出真正回复
        )
        
        # 创建对话链，将提示词模板和输出解析器结合起来。
        self.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True, #是否打印详细信息
            output_parser=output_parser
        )

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
        if self.chain is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model")
        
        # 获取历史记录
        history = self.memory.load_memory_variables({})["history"]
        
        # 生成回复
        result = self.chain.predict(history=history, input=input_text)
        
        # 保存到记忆
        self.memory.save_context({"input": input_text}, {"output": result["response"]})
        
        return result["response"]

model_manager = ModelManager()



