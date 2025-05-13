import re
def split_raw_by_chapter(txt_path:str): #将小说内容按章节进行分割 
    with open(txt_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    # raw_text = re.sub(  #将每回末尾'* * *'开头的以到下一个章节标题之间的注释内容删除
    #     r'\*\s*\*\s*\*.*?(?=(第[一二三四五六七八九十百千〇零\d]+回\s+[^ ]|$))',
    #     '',
    #     raw_text,
    #     flags=re.DOTALL
    # )
    #下面的正则可以匹配“第x章/篇/回/卷//部编/”和“卷x”和“篇x”，作为章节分割的公式
    chapters = re.split(r'(?=(?:第[一二三四五六七八九十百千〇零\d]+[章篇回卷部编]\s+[^ \n]{2,})|(?:卷[一二三四五六七八九十百千〇零\d]+\s*[^ \n]{2,})|(?:篇[一二三四五六七八九十百千〇零\d]+\s*[^ \n]{2,}))', raw_text) #按章节进行分割
    # if chapters and chapters[0].strip() == '':
    #     chapters = chapters[1:]
    return chapters

import json
def spilt_chapters_by_amount(chapters:list,dirty_chunks_save_path:str): #对每个章节按500字数的标准进行分割，但是会保留完整句子，不是直接切割。
    chunks = []
    for chapter in chapters:
        chunk = ""
        sentences = re.split(r'(?<=[。！？”])\s*', chapter) #将每章内容按句子结束标准进行分割 
        for s in sentences:  #将每句话组合在一起，限制为500字
            if len(chunk) + len(s) < 500:
                chunk += s
            else:
                chunks.append(chunk)
                chunk = s
        if chunk: #加入最后一个chunk
            chunks.append(chunk)
    with open(dirty_chunks_save_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def data_cleaning(dirty_chunks_data_path:str,cleaned_chunks_save_path:str): #数据清理
    with open(dirty_chunks_data_path, 'r', encoding='utf-8') as f:
        dirty_data = json.load(f)
    cleaned_chunks = [re.sub(r'\[\d+\]', '', chunk) for chunk in dirty_data] #将‘[1]’这样的注脚去除
    cleaned_chunks = [re.sub(r'\s+', ' ', chunk).strip() for chunk in cleaned_chunks] #去除多于的空格、换行符等等
    with open(cleaned_chunks_save_path,'w',encoding='utf-8') as f:
        json.dump(cleaned_chunks, f, ensure_ascii=False, indent=2)

def delete_empty_dic(formatted_data_path:str,train_data_save_path:str): #将空的对话数据删除
    with open(formatted_data_path,'r',encoding='utf-8') as f:
        formatted_data = json.load(f)
    formatted_data = [item for item in formatted_data if item["conversations"]!=[]]
    with open(train_data_save_path,'w',encoding='utf-8') as f:
        json.dump(formatted_data,f,ensure_ascii=False,indent=2)

