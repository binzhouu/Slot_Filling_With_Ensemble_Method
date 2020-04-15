## Introduction
很多论文中会将intent和slot进行绑定，用联合模型进行处理，本仓库以intent作为输入，加上额外的辞典属性特征，
并以sequence labeling为基础的ensemble方案，进行模型结果融合，用集成方案减小单个模型的误差，并可以和规
则模版产生的结果再次集成。

## Features
- char、word的embedding；
- 辞典特征
- intent特征

## Model Structure

## Slot Filling

### 路径：models/slot_filling/inference/inference_by_bilstm.py

- 参数说明:  
    text: 用户的input_text; 类型：str  
    intent: 当前input_text经ensemble模块后给出的意图; 类型：str  
    session_keep: 当前对话是否为多轮; 类型：bool  
    previous_intent: 上一轮对话的意图; 类型：str  
    
- return:  
    slot: 当前input_text所包含的槽位  

- 示例:  
    ####
        from models.slot_filling.inference.inference_by_bilstm import SlotFilling
        texts = ['打开灯的灯']
        intents = ['turn_on']
        slot_filling = SlotFilling()
        for t, i in zip(texts, intents):
            res = slot_filling.inference(t, i, False, None)
            print('text: %s, res: %s' % (t, res))