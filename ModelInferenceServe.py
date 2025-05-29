from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.output_parsers import RegexParser
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import os
# from llama_cpp import Llama

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chain = None
        self.memory = None
        self.vector_store = None
        self.embeddings = None
    
    def load_model(self, model_path:str):
        if model_path == self.model_path:
            return #模型路径相同，不重新加载
        print("切换模型为",model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(f"model_file/finetuned_model/{model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(f"model_file/finetuned_model/{model_path}")
        # self.tokenizer = AutoTokenizer.from_pretrained(f"model_file/finetuned_model/Qwen2.5-0.5B-Instruct")
        # self.model = AutoModelForCausalLM.from_pretrained(f"model_file/finetuned_model/Qwen2.5-0.5B-Instruct")
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
        
        # 创建对话模板，加入检索到的相关文档
        template = """你是一个中文对话助手，请根据用户输入和检索到的相关文档生成简洁、自然、有逻辑的中文回答，一定不要重复用户的输入，一定不要输出英文，并且只需给出一次回答。
            检索到的相关文档：
            {context}

            当前对话历史：
            {history}

            用户输入：{input}
            AI回复："""
        
        prompt = PromptTemplate(
            input_variables=["context", "history", "input"], #输入变量，用于替换模板中的占位符。
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

        # 加载文档到向量存储
        try:
            print("正在加载文档到向量存储...")
            self.load_documents("novels")  # 使用novels目录作为文档源
            print("文档加载完成")
        except Exception as e:
            print(f"文档加载失败: {str(e)}")
            print("系统将继续运行，但将无法使用文档检索功能")

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

    def load_documents(self, documents_dir: str):
        """加载文档并创建向量存储"""
        # 初始化文本分割器，使用更合理的chunk大小
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 增加chunk大小，保持更多上下文
            chunk_overlap=200,  # 增加重叠，保持上下文连贯性
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 添加中文分隔符
        )
        
        # 加载文档
        loader = DirectoryLoader(
            documents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # 分割文档
        texts = text_splitter.split_documents(documents)
        print(f"文档已分割为 {len(texts)} 个片段")
        
        # 初始化embeddings模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./bge-large-zh-v1.5",  # 使用中文模型
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # 创建向量存储
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print(f"已加载 {len(texts)} 个文档片段到向量存储")

    def generate_response(self, input_text:str) -> str:
        if self.chain is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model")
        
        # 获取历史记录
        history = self.memory.load_memory_variables({})["history"]
        
        # 检索相关文档
        context = ""
        if self.vector_store is not None:
            try:
                # 使用相似度搜索
                docs = self.vector_store.similarity_search(
                    input_text,
                    k=3  # 获取最相关的3个文档
                )
                
                # 打印检索到的文档
                print("\n检索到的相关文档片段：")
                for i, doc in enumerate(docs, 1):
                    print(f"\n文档 {i}:")
                    print(f"内容: {doc.page_content[:300]}...")  # 显示更多内容
                
                # 使用检索到的文档
                context = "\n".join([doc.page_content for doc in docs])
                
                if not context:
                    print("警告：没有找到相关文档片段")
            except Exception as e:
                print(f"文档检索出错: {str(e)}")
        
        # 生成回复
        result = self.chain.predict(context=context, history=history, input=input_text)
        
        # 保存到记忆
        self.memory.save_context({"input": input_text}, {"output": result["response"]})
        
        return result["response"]

model_manager = ModelManager()