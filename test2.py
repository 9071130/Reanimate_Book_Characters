import json
broken_json = '''
[
  {
    "role": "user",
    "content": "你好"
  }
  {
    "role": "gpt",
    "content": "你好，有什么可以帮助你的？"
  }
]
'''
try:
    json.loads(broken_json)
except Exception as e:
    print("格式出错",{e})