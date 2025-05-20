# 项目简介

本项目致力于实现小说的 “虚拟角色复活”，用户只需上传一本小说，并选择其中任意一个角色，系统便可自动生成一个微调后的语言模型。该模型不仅能够复现该角色的语言风格，还具备小说中的背景知识与人物设定，实现自然、拟人的对话体验。项目涵盖从数据集构建、模型微调到部署交互的完整流程，为个性化AI角色提供了低门槛、高还原度的解决方案。

# 使用方法

项目分为前端部分和后端部分。

前端地址:https://github.com/9071130/Reanimate_Book_Characters_Frontend.git

后端地址:https://github.com/9071130/Reanimate_Book_Characters_Backend.git

视频教程:https://www.bilibili.com/video/BV1d6EXz6Eyk/?vd_source=3e7717db1fb71d3afc5e9af471a52848

### 第一步，运行后端

强烈推荐使用linux去跑，可以省去很多环境问题，下面以租用AutoDL平台的显卡运行作为例子。

1.租用设备

网址:https://www.autodl.com

打开网站后点击算力市场，选择任意地区(建议西北，便宜)，选择4090，点击租用。环境选择如下图所示，之后点击创建并开机。

![图片](https://github.com/user-attachments/assets/ca37f7c3-a09c-4560-af01-43776263955f)

2.利用vscode进行SSH连接

在vocde扩展商店中下载Remote - SSH，之后选择新建远程。

![图片](https://github.com/user-attachments/assets/db84d8e8-6683-40f5-bffb-4abe953d32cc)

输入AutoDL给你的账号密码。

![图片](https://github.com/user-attachments/assets/a38dbb78-b2fc-4456-9c88-d28758544234)


进入SSH后，打开文件夹autodl-temp，这是在数据盘的文件夹，后面如果可以扩容，更方便。

![图片](https://github.com/user-attachments/assets/70e79b99-373f-4558-84ef-6f625a205805)

3.克隆项目，安装依赖

进入文件夹后，打开终端，输入命令`conda init`,再删除当前终端，重新打开。

![图片](https://github.com/user-attachments/assets/1e839aa7-2b99-4fb7-ac2c-2aa31b80cfe8)

重新打开终端后，会发现当前处在一个叫base的conda环境中，输入以下命令:

`git clone https://github.com/9071130/Reanimate_Book_Characters_Backend.git`

进入文件夹:`cd Reanimate_Book_Characters_Backend`

安装依赖:`pip install -r requirements.txt`

切换HuggingFace镜像源:`export HF_ENDPOINT=https://hf-mirror.com`

启动项目:`uvicorn Api:APP --reload --host 0.0.0.0 --port 8001`，如果看到如下页面就说明后端服务启动成功了。

![图片](https://github.com/user-attachments/assets/4c5faded-dab8-41ab-be41-f9427b02d668)

### 第二步，运行前端

在本机新建一个文件夹，用vscode打开，在终端输入以下命令:

`git clone https://github.com/9071130/Reanimate_Book_Characters_Frontend.git`

进入项目文件夹:`cd Reanimate_Book_Characters_Frontend`

安装依赖:`npm install`

运行项目:`npm run dev`,看到如下提示就说明前端项目启动完成了，之后就像视频一样操作就可以了。

![图片](https://github.com/user-attachments/assets/1bdd49af-fa9e-43af-b6b7-901d23e21f8d)

# 效果演示

我用了大概2800条的对话数据，训练了三轮，基座模型是Qwen2.5-3B，感觉效果还不错。

![图片](https://github.com/user-attachments/assets/33b8902b-905a-4304-b7e2-eb44178f0b82)