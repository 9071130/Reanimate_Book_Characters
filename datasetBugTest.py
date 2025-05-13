import subprocess
subprocess.Popen([ #启动数据集制作子进程
    "python","DeepSeekServe.py",
    "--book_name",'西游记',
    "--role_name",'唐僧',
    "--cleaned_chunks_data_path","output_text_data/cleaned_chunks_data.json",
    "--formatted_cleaned_data_save_path","output_text_data/formatted_cleaned_data.json",
    "--api_key",'sk-49aaa14d7a9248ad84bc58461cc961e4',
    "--task_id",'123test'
])