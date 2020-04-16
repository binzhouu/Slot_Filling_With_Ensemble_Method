## Introduction
很多论文中会将intent和slot进行绑定，用联合模型进行处理，本仓库以intent作为输入，加上额外的辞典属性特征，
并以sequence labeling为基础的ensemble方案，进行模型结果融合，用集成方案减小单个模型的误差，并可以和规
则模版产生的结果再次集成。

## Features
- 基本的char、word的embedding；
- 辞典特征  
辞典配置路径：configs/specific_words.txt
特征处理方式：
    - 假设有5类字典：[lexi1]、[lexi2]、[lexi3]、[lexi4]、[lexi1];  
    - 句子中某一个word属于[lexi1]和[lexi5]，用one-hot形式呈现：[1,0,0,0,1],再把one-hot特征抽象到一个embedding的类别;  
    - 这样处理的好处是模型能够学到词的属性特征，尤其当槽位包含辞典中的生词时，我们不需要更新模型，也可以取得相应的slot；
- intent特征  
意图特征，作为模型输入，不同意图会和相应槽位做强绑定;  

## Model Structure
- 路径：model/  
- 字符级别的模型：model/bilstm_crf.py & model/multiply_bilstm_crf.py  
- 字符 + 词 + 手工特征的模型：model/cnn_attn_lstm_crf.py  
这里简单说一下处理方式：  
    - 字符向量经CNN抽取特征；  
    - intent特征会再dim=1的维度，做repeat，扩充到字符维度；  
    - 字符特征与intent特征做拼接，得到字符级别特征；  
    - intent特征会再dim=1的维度， 做repeat，扩充到字符维度；  
    - 词特征、辞典特征、intent特征做拼接，得到词级别特征；  
    - 借鉴self_attention思路，构建q,k,v矩阵，将字符级别特征与词级别特征做映射；  
    - attn_output经双向lstm，抽取语义特征，再由CRF做编码，输出结果  

## Sequence Labeling Ensemble
理论说明：  
我们之所以用类似HMM、条件随机场这类都无向图模型解序列问题，是因为结果受到转移矩阵的约束，如果我们用常规的ensemble思路去
对不同模型结果做Votes，或计算confidence，都不能保证序列都合理性；因此如果想沿用无向图都思路，用viterbi解码，我们需要
手动的提取emission_score和transition_score，就可以很好融合多个序列标注模型的特性，并且其结果也一定是合理序列。  
- 另外，该方法可以和pattern得到的结果再次进行融合，以提升最终结果；  

## 集成模型推理：
- 路径：ensemble_method/slot_ensemble.py  

## 单个模型推理：
- 路径：inference/  

- 参数说明:  
    text: 用户的input_text; 类型：str  
    intent: 当前意图; 类型：str  
    session_keep: 当前对话是否为多轮; 类型：bool  
    previous_intent: 上一轮对话的意图; 类型：str  
    
- return:  
    slot: 当前input_text所包含的槽位  

- 示例:  
    ####
        texts = ['我想听月亮下的人']
        intents = ['play_music']
        slot_filling = SlotFilling()
        for t, i in zip(texts, intents):
            res = slot_filling.inference(t, i, False, None)
            print('text: %s, res: %s' % (t, res))

## 模型训练：
- 路径：run.py  
- 参数配置：conf/model.yaml & conf/data.yaml
只需要更改参数model_num，就可以选择对应的模型结构；  

##
有任何问题，欢迎和我联系。