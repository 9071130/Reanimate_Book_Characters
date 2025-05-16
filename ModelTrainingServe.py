import argparse
from unsloth import FastLanguageModel #加载模型的工具
import json,os,time
from datasets import Dataset
from trl import SFTTrainer  # 导入 SFTTrainer，用于监督式微调
from transformers import TrainingArguments  # 导入 TrainingArguments，用于设置训练参数
from unsloth import is_bfloat16_supported  # 导入函数，检查是否支持 bfloat16 数据格式
import math

def start_training_model(pretrained_model_path,train_data_path,spliced_data_save_path,hf_api,upload_path_safetensors,upload_path_gguf,book_name,role_name,task_id):
    # 安装 unsloth 包。unsloth 是一个用于微调大型语言模型（LLM）的工具，可以让模型运行更快、占用更少内存。
    # !pip install unsloth
    # 安装 bitsandbytes 和 unsloth_zoo 包。
    # !pip install bitsandbytes unsloth_zoo
    max_seq_length = 2048 #一次能处理的最大token长度
    dtype = None #设置数据类型，让unsloth自动选择合适的计算精度
    load_in_4bit = True #使用4bit量化，节省内存
    #加载预训练模型，并获取分词器即将文本转为token的工具，模型只能识别token
    model, tokenizer= FastLanguageModel.from_pretrained(
        # model_name = "model_file/pretrained_model/DeepSeek-R1-Distill-Qwen-1.5B",
        model_name = pretrained_model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )

    # 定义一个用于格式化提示的多行字符串模板
    train_prompt="""以下是书籍《{book_name}》中“{role_name}”与他人的对话，请你模仿{role_name}的说话风格，继续与用户进行交流。
    {chat_history}
    ###{role_name}的回复:
    {response}
    """
    EOS_TOKEN = tokenizer.eos_token #文本结束标志，必须添加结束标记
    with open(train_data_path,"r",encoding='utf-8') as f: #拼接数据
        conversions_data = json.load(f)
        texts = [] #拼接后的数据
        for conversion in conversions_data:
            chat_history = ''
            conversations = conversion.get("conversations", [])# 判断是否存在conversations字段且为list
            if not isinstance(conversations, list):
                continue  # 跳过非正常结构
            for sentence in conversations:
                if not isinstance(sentence, dict):
                    continue
                role = sentence.get("role", "")
                content = sentence.get("content", "")
                if not isinstance(content, str):
                    continue

                current_role  = role_name if role == "gpt" else "其他角色"
                if current_role  == role_name and len(chat_history) > 1:
                    text = train_prompt.format(book_name=book_name,role_name=role_name,chat_history=chat_history,response=content)+EOS_TOKEN
                    texts.append(text)
                chat_history += f"{current_role}:{content}\n"
        dataset = {"text":texts}
    with open(spliced_data_save_path,"w",encoding='utf-8') as f: #保存拼接好的数据后续可以查看
        json.dump(dataset,f,ensure_ascii=False,indent=2)

    #将数据集转换成dataset格式
    dataset = Dataset.from_dict(dataset)
    dataset_length = len(dataset)

    #将模型设置为训练模式，开启梯度计算，推理模式下梯度计算是关闭的
    FastLanguageModel.for_training(model)
    #用 LoRA 技术对预训练模型进行参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）。即在不训练整个模型的情况下，通过 LoRA 技术仅调整部分关键层，从而降低计算成本和显存占用。
    model = FastLanguageModel.get_peft_model(
        model,  # 传入已经加载好的预训练模型
        r = 16,  # 设置 LoRA 的秩，决定添加的可训练参数数量，一般 r 取 8、16 或 32。
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # 指定模型中需要微调的关键层，q_proj, k_proj, v_proj, o_proj：影响注意力机制（Query、Key、Value、Output）。
                        "gate_proj", "up_proj", "down_proj"],
        # lora_alpha 较小,LoRA 对原始模型的影响小，模型保留更多原始特性。适用于小数据集或微调不希望改变太多的情况。
        # lora_alpha 较大（例如 32 或 64）：LoRA 影响力增加，使微调的效果更显著。适用于大数据集，希望模型适应新任务的情况。
        lora_alpha = 64,
        lora_dropout = 0,  # 设置防止过拟合的参数，这里设置为 0 表示不丢弃任何参数
        bias = "none",    # 设置是否添加偏置项，这里设置为 "none" 表示不添加
    #     梯度检查点（Gradient Checkpointing） 是一种显存优化技术：
    #     传统训练会存储所有层的计算结果，占用大量显存。
    #     启用 Gradient Checkpointing，仅存储关键层计算结果，其他部分在反向传播时重新计算，从而减少显存占用。
    # "unsloth" 代表使用 Unsloth 库 提供的高效实现。
        use_gradient_checkpointing = "unsloth",  # 使用优化技术节省显存并支持更大的批量大小
        random_state = 3407,  # 设置随机种子，确保每次运行代码时模型的初始化方式相同
        use_rslora = False,  # Rank Stabilized LoRA（RS-LoRA） 是一种优化 LoRA 训练的技术，通常用于 LoRA 训练不稳定的情况。
    # LoftQ（Low-Rank Fine-Tuning Quantization） 是一种结合 LoRA 和量化的方法。
    # 这里设为 None，表示 不启用量化，保持模型原始精度。
        loftq_config = None,
    )

    trainer = SFTTrainer(  # 创建一个 SFTTrainer 实例
        model=model,  # 传入要微调的模型
        tokenizer=tokenizer,  # 传入 tokenizer，用于处理文本数据
        train_dataset=dataset,  # 传入训练数据集
        dataset_text_field="text",  # 指定数据集中文本字段的名称
        max_seq_length=max_seq_length,  # 设置最大序列长度
        dataset_num_proc=8,  # 设置数据处理的并行进程数
        packing=False,  # 是否启用打包功能（这里设置为 False，打包可以让训练更快，但可能影响效果）
        args=TrainingArguments(  # 定义训练参数
            per_device_train_batch_size=32,  # 每个设备一次处理/向前传播的数据条数
            gradient_accumulation_steps=2,  # 累积多少个小btach后再进行反向传播更新梯度，总的batch_size就是上下两个数相乘
            warmup_steps=5,  # 预热步数，训练开始时学习率逐渐增加的步数
            max_steps=math.ceil(dataset_length*2/64),  # 最大训练步数
            learning_rate=2e-4,  # 学习率，模型学习新知识的速度
            fp16=not is_bfloat16_supported(),  # 是否使用 fp16 格式加速训练（如果环境不支持 bfloat16）
            bf16=is_bfloat16_supported(),  # 是否使用 bfloat16 格式加速训练（如果环境支持）
            logging_steps=10,  # 每隔多少步记录一次训练日志
            optim="adamw_torch",  # 使用的优化器，用于调整模型参数
            weight_decay=0.01,  # 权重衰减，防止模型过拟合
            lr_scheduler_type="linear",  # 学习率调度器类型，控制学习率的变化方式
            seed=3407,  # 随机种子，确保训练结果可复现
            output_dir="outputs",  # 训练结果保存的目录
            report_to="none",  # 是否将训练结果报告到外部工具（如 WandB），这里设置为不报告
        ),
    )

    trainer_stats = trainer.train()#开始训练

    # # 将微调后的模型保存为safetensors和GGUF格式
    # HUGGINGFACE_TOKEN = hf_api #huggingface的api令牌
    # model.push_to_hub_merged(upload_path_safetensors, tokenizer, save_method = "merged_16bit", token = HUGGINGFACE_TOKEN,safe_serialization = True) #保存为safetensors格式，并推送至hf
    # model.push_to_hub_gguf(upload_path_gguf, tokenizer, token = HUGGINGFACE_TOKEN) #保存为gguf格式，并推送至hf

    #将模型保存到本地用于后续推理使用
    os.makedirs(f"model_file/finetuned_model/{task_id}")
    model.save_pretrained_merged(f"model_file/finetuned_model/{task_id}",tokenizer, save_method = "merged_16bit")

def write_status(task_id,status): #保存任务状态供前端查询
    os.makedirs("model_tasks_status",exist_ok=True)
    status_path = f"model_tasks_status/{task_id}.json"
    try:
        if os.path.exists(status_path):
            with open(status_path,'r',encoding='utf-8') as f:
                status_list = json.load(f)
        else:
            status_list = []
        #将任务状态和时间都保存
        status_list.append({"status":status,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")})
        #写回文件
        with open(status_path,'w',encoding='utf-8') as f:
            json.dump(status_list,f,ensure_ascii=False,indent=2)
    except Exception as e:
        print(f"写入任务状态时发生错误:{e}")
if __name__ == "__main__":
    #接收训练的相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', default=None)
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--spliced_data_save_path',type=str, default=None) #拼接了train prompt后的数据
    parser.add_argument('--hf_api', default=None)
    parser.add_argument('--upload_path_safetensors', default=None)
    parser.add_argument('--book_name',type=str, default=None)
    parser.add_argument('--role_name',type=str, default=None)
    parser.add_argument('--task_id',type=str, default=None)
    parser.add_argument('--upload_path_gguf', default=None)
    args = parser.parse_args()
    pretrained_model_path = args.pretrained_model_path
    train_data_path = args.train_data_path
    spliced_data_save_path = args.spliced_data_save_path
    hf_api = args.hf_api
    upload_path_safetensors = args.upload_path_safetensors
    upload_path_gguf = args.upload_path_gguf
    book_name =  args.book_name
    role_name = args.role_name
    task_id = args.task_id
    write_status(task_id,"running")
    start_training_model(pretrained_model_path,train_data_path,spliced_data_save_path,hf_api,upload_path_safetensors,upload_path_gguf,book_name,role_name,task_id)
    write_status(task_id,"success")

