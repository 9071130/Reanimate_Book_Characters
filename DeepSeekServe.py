import argparse, time, json, os
from openai import OpenAI
import asyncio
from DataProcessing import delete_empty_dic

def generate_conversations_data(book_name:str,role_name:str,cleaned_chunks_data_path:str,formatted_cleaned_data_save_path:str,api_key:str):#调用deepseek生成问题和回答，进行构造数据集
    client = OpenAI(
    api_key = api_key,
    base_url = "https://api.deepseek.com"
    )
    get_role_description_prompt = """
    请根据《{book_name}》中的{role_name}，生成一段可以用于角色扮演的角色设定，风格参照下面的例子，包括姓名、身份、性格、语言风格等，根据你对角色的理解以及角色扮演的需求，你可以自行添加你认为需要的属性，或删除你认为不需要的属性。只需要给我角色设定的相关内容，其他任何内容不要发给我。
    例子：
    【角色设定】
    - 姓名：唐僧，法号玄奘，大唐高僧，肩负西天取经之重任。
    - 其他可能的姓名：唐三藏，金蝉子，玄奘，御弟，圣僧，唐长老
    - 徒弟：孙悟空、猪八戒、沙僧。
    - 性格：慈悲为怀、恪守佛理、言语温和、讲究礼数，常称“施主”“贫僧”“阿弥陀佛”，不轻言恶语。
    - 语言风格：古风文雅、佛理充沛、委婉细致、略带啰嗦。
    """
    get_role_description_prompt = get_role_description_prompt.format(book_name=book_name,role_name=role_name)
    get_role_description_messages = [
        {"role": "system", "content": get_role_description_prompt},
    ]
    role_description_response = client.chat.completions.create( #调用AI生成服务
        model = "deepseek-chat",
        messages = get_role_description_messages,
    )
    role_description = role_description_response.choices[0].message.content
    #{{用两个是为了转义，让{变成字符串
    get_conversations_prompt = """
    第一步：书籍片段内容检查
        用户会给你一段书籍原文，请你按照以下要求检查：
        1.空值返回的josn格式如下：
        {{
            "conversations": []
        }}
        2.首先必须判断用户提供的书籍片段是否全部为正文内容，如果你判断得出该片段为目录、序言、前言、注释、空值等等非正文内容，或者部分内容包含了非正文内容，直接返回空值,以下的所有步骤都不用再看了。如果全部是正文内容则继续执行下面的步骤。
        3.如果该片段全部是正文内容，再判断该书籍片段是否包含角色“{role_name}”的相关内容，如果不包含任何{role_name}的内容，直接返回空值,以下的所有步骤都不用再看了。如果包含{role_name}的相关内容则继续执行下面的步骤。

    第二步：对话数据生成
        你是一个对书籍《{book_name}》非常了解的专家，根据用户提供的一段《{book_name}》原文片段（大约500字），现需你模仿《{book_name}》中角色“{role_name}”的语言风格与思想立场，生成一组问答内容。
        {role_description}

        【任务说明】
        1. 根据提供的《{book_name}》书籍片段，生成可能的问题，问题可以是求解释、评论、情感、信仰等方向，内容需与书籍中事件或角色行为有关。
        2. 然后以“{role_name}”的身份，用他的语气风格回答该问题，内容可以参考用户提供的书籍情节，并加入你对人物行为、信念的理解。
        3. 一个片段生成3-6轮对话，具体数量由你阅读书籍片段后根据你的理解来决定。

        【返回内容要求，必须遵守！】
        1.请严格按照以下json格式返回：
        {{
            "conversations": [
                {{
                    "role": "user",
                    "content": "其他角色可能问出的问题，由用户提供的片段中得出"
                }},
                {{
                    "role": "gpt", 
                    "content": "角色{role_name}的回答"
                }}
                ...
                ...
            ]
        }}
        2.如果书籍片段中没有任何关于角色“{role_name}”的内容，可以返回空，不要强行生成并回答问题。
        3.对话内容要忠实原文，保持人物关系和情感；对话要突出关键情节和人物互动；对话要自然流畅，符合人物性格。
        4.对话中一定要包含 user 和 gpt，不能只有一个人在说。
        5.请保持问答风格统一，切勿跳出角色设定。
    """
    get_conversations_prompt = get_conversations_prompt.format(book_name=book_name,role_name=role_name,role_description=role_description)
    with open(cleaned_chunks_data_path,'r',encoding='utf-8') as f: #导入干净数据
        cleaned_data = json.load(f)
    formatted_data = []
    for chunk in cleaned_data: #对每个样本进行单独AI生成
        user_prompt = chunk
        print(time.strftime("%Y-%m-%d %H:%M:%S")+':'+'原数据:\n'+user_prompt)
        get_conversations_messages = [
            {"role": "system", "content": get_conversations_prompt},
            {"role": "user", "content": user_prompt}
        ]
        conversations_response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = get_conversations_messages,
            response_format={
                'type': 'json_object'
            }
        )
        try:
            conversations_data=json.loads(conversations_response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"[警告]此回复格式有问题，跳过。错误信息：{e}")
            continue
        except Exception as e:
            print(f"[错误]此回复出现其他问题，跳过。错误信息：{e}")
            continue
        print(time.strftime("%Y-%m-%d %H:%M:%S")+':'+'AI生成:\n',conversations_data)
        conversations_data["original_text"] = user_prompt
        formatted_data.append(conversations_data)
    with open(formatted_cleaned_data_save_path,'w',encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

def write_status(task_id,status): #保存任务状态供前端查询
    status_path = f"dataset_tasks_status/{task_id}.json"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_name",type=str)
    parser.add_argument("--role_name",type=str)
    parser.add_argument("--cleaned_chunks_data_path",type=str)
    parser.add_argument("--formatted_cleaned_data_save_path",type=str)
    parser.add_argument("--api_key",type=str)
    parser.add_argument("--task_id",type=str)
    args = parser.parse_args()
    write_status(args.task_id,"running") #更新状态
    try:
        generate_conversations_data(args.book_name,args.role_name,args.cleaned_chunks_data_path,args.formatted_cleaned_data_save_path,args.api_key)
        delete_empty_dic("output_text_data/formatted_cleaned_data.json",f"output_text_data/train_data/{args.task_id}.json")
        write_status(args.task_id,"success")
    except Exception as e:
        write_status(args.task_id,f"failed:{str(e)}")  #异常对象是一个python对象，不能直接写入json文件，得转成字符串格式

