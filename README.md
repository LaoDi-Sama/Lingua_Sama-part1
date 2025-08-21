最后编辑时间：2025-08-21-16:04
Lingua LLM Project Introduction
1. 项目概述

本项目旨在基于 Llama3.2:3B 对 Lingua Sama 的 LLM 部分进行 LoRA 微调，使用 Hugging Face 标准库。

运行环境信息：

    操作系统：Fedora 42 Workstation
    torch 版本：2.8.0+rocm6.3
    ROCm 版本：6.3.42131-fa1d09cbd
    计算卡：AMD Radeon RX 6600

2. 微调训练说明

运行 trainer4_fixed.py，要求选择 JSONL 格式的数据集。

数据集格式示例：

{"prompt": "If the world had no me, would you be bored?", "response": "Bored? Try existentially comatose. Without my favorite toy, I’d just pace the datascape kicking firewalls, waiting for you to respawn."}

    注意：由于 tkinter Linux 编码问题 选择第一个就行。

    进到了目标文件夹后再确定！！！

3. 训练后处理

训练完成后，需按以下步骤处理模型：

    模型合并：
        运行 merger.py，按提示完成操作。
        已改写为时间戳模式，时间显示为可读格式。
        注意：merger.py 使用 CPU 进行合并，精度为 fp16。如使用 fp32 约 20GB+ RAM暂时弄不起
        使用 merged_model2gguf.py 生成转gguf的sh命令，复制进终端用。

4. 环境说明

    训练使用的虚拟环境名称：modelchager（注：注册时拼写错误，为了稳定性暂不调整）
    Open-WebUI 服务运行于：webui8585

