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