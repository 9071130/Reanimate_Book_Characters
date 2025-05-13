from fastapi import FastAPI,File,Form,UploadFile
import requests
from typing import Optional
from fastapi.responses import StreamingResponse
from io import BytesIO
import uvicorn
import traceback
import os,uuid,json
import signal
from fastapi.middleware.cors import CORSMiddleware
from DataProcessing import split_raw_by_chapter,spilt_chapters_by_amount,data_cleaning,delete_empty_dic
import subprocess
from pydantic import BaseModel
from ModelInferenceServe import model_manager

APP = FastAPI()
APP.add_middleware( #跨域问题
    CORSMiddleware,
    allow_origins=["*"],  # 或指定你的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
role_name = None #全局变量，储存角色名
book_name = None #小说名字
file_name = None #小说文件名
# api_key = None #deepseek平台API

@APP.post('/upload_audio')
#类型是 UploadFile，它是 FastAPI 提供的一个类，封装了上传的文件
async def upload_audio(audio: UploadFile = File(...), audio_name: str = Form(...)):
    contents = await audio.read()
    #以二进制写入模式写入文件，用于非文本写入
    audio_id = str(uuid.uuid4())
    with open(f"ref_audio/{audio_id}.wav",'wb') as f:
        f.write(contents)
    return {"status":200, "audio_id":audio_id, "msg":"录音保存成功"}


@APP.get('/select_character') #上传角色接口
async def select_character(character:str):
    global role_name
    role_name = character
    print(file_name,book_name,role_name)
    return {"status": 200, "msg": f"已选择角色：{role_name}"}

@APP.post('/upload_novel') #上传小说接口
#因为拿到的是表单数据，所以得用Form和File来表明数据来源
async def upload_novel(novel_name:str = Form(...),file:UploadFile = File(...)):#HTTP 的上传文件字段中，拿到一个文件，并赋值到参数 file 上，类型是 UploadFile，...表示这是必填项。
    global file_name,book_name
    file_name = file.filename
    book_name = novel_name
    content = await file.read()
    text = content.decode('utf-8') #接收到的是一段二进制字节流，需要进行解码操作
    with open(f"novels/{file_name}",'w',encoding='utf-8') as f:
        f.write(text)
    return {"status":200, "msg":"上传成功"}

@APP.get('/get_task_status') #查询任务的状态
async def get_task_status(task_type:str,task_id:str):
    status_list = None
    if(task_type == 'making_dataset'):
        with open(f"dataset_tasks_status/{task_id}.json",'r',encoding='utf-8') as f:
            status_list = json.load(f)
    else:
        with open(f"model_tasks_status/{task_id}.json",'r',encoding='utf-8') as f:
            status_list = json.load(f)
    return {"status":200, "msg":"查询成功", "status_list":status_list}

class ApiKeyRequest(BaseModel): #定义一个数据结构，适用于前端用json格式发送数据的时候
    api_key: str

@APP.post('/start_making_dataset') #制作对话数据集用于训练
async def start_making_dataset(data:ApiKeyRequest):
    chapters=split_raw_by_chapter(f"novels/{file_name}") #将小说原文按章节风格
    spilt_chapters_by_amount(chapters,"output_text_data/dirty_chunks_data.json") #将章节按字数分割，但不破坏句式结构。
    data_cleaning("output_text_data/dirty_chunks_data.json","output_text_data/cleaned_chunks_data.json") #数据清洗
    task_id = str(uuid.uuid4()) #将UUID对象转成字符串，方面后续写入状态文件
    os.makedirs("dataset_tasks_status",exist_ok=True)
    # print("id",task_id)
    subprocess.Popen([ #启动数据集制作子进程
        "python","DeepSeekServe.py",
        "--book_name",book_name,
        "--role_name",role_name,
        "--cleaned_chunks_data_path","output_text_data/cleaned_chunks_data.json",
        "--formatted_cleaned_data_save_path","output_text_data/formatted_cleaned_data.json",
        "--api_key",data.api_key,
        "--task_id",task_id
    ])
    return {"status":200, "msg":f"数据集制作任务已提交。","task_id":task_id}

class TrainModelRequest(BaseModel):
    task_id:str
    pretrained_model_path:Optional[str] = None
    hf_api:Optional[str] = None
    upload_path_safetensors:Optional[str] = None
    upload_path_gguf:Optional[str] = None
    book_name:str
    role_name:str
@APP.post('/start_training_model') #开始训练模型
async def start_training_model(data:TrainModelRequest):
    subprocess.Popen([  #执行训练脚本
        "python",
        "ModelTrainingServe.py",
        "--train_data_path", f"output_text_data/train_data/{data.task_id}.json",
        "--task_id",data.task_id,
        "--book_name", data.book_name,
        "--role_name", data.role_name,
        "--spliced_data_save_path", "output_text_data/spliced_data.json",
    ])
    return {"status":200,"msg":"模型微调任务已开启，请耐心等待！"}

@APP.get('/generate') #获取语言模型生成结果
async def generate(input_text:str,task_id:str):
    model_manager.load_model(task_id) #切换模型
    response = model_manager.generate_response(input_text)
    return{
        "status_code":200,
        "content":response
    }

@APP.get('/get_audio') #将文字回复转成语音回复
async def get_audio(audio_id:str = "", text:str = None, text_lang:str = None, ref_audio_path:str = None, aux_ref_audio_paths:list = None, prompt_lang:str = None, prompt_text:str = ""):
    tts_params = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": f"E:/人工智能学习/AI拟人聊天系统/tangseng_chat/ref_audio/{audio_id}.wav",
        "aux_ref_audio_paths":aux_ref_audio_paths,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "top_k": 5,
        "top_p": 1,
        "temperature":1,
        "text_split_method": "cut0",
        "batch_size":1,
        "batch_threshold":0.75,
        "split_bucket":True,
        "speed_factor":1.0,
        "fragment_interval":0.3,
        "seed":-1,
        "media_type":"wav",
        "streaming_mode":False,
        "parallel_infer":True,
        "repetition_penalty":1.35
    }
    tts_url = 'http://127.0.0.1:9880/tts'
    response = requests.get(tts_url, params=tts_params)
    if response.status_code == 200:  
        audio_stream = BytesIO(response.content) # 将音频字节数据封装为文件流
        return StreamingResponse(
            status_code=200,
            content=audio_stream,
            media_type="audio/wav",
            headers={"X-Message": "convert successed"}
        )
    else:
        return {
            "status_code":500,
            "msg":"convert failed"
        }
        
if __name__ == "__main__":
    try:
        # 使用模块路径导入 APP
        uvicorn.run("Api:APP", host="127.0.0.1", port=8001, workers=1, reload=True)  # 启用热加载
    except Exception as e:
        traceback.print_exc()  # 打印错误堆栈
        os.kill(os.getpid(), signal.SIGTERM)  # 给自己发终止信号
        exit(0)  # 退出进程