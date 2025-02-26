# 基于ONNXRuntime推理框架的通用信息抽取

本项目通过使用[文本信息抽取UIE模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/applications/information_extraction/text/README.md)和[文档信息抽取UIE-X模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/applications/information_extraction/document/README.md)进行通用信息提取，结合`ONNXRuntime`推理引擎，有效解决了依赖多、部署复杂的问题。通过将模型转换为`ONNX`格式，并利用`ONNXRuntime`进行高效推理，可以在各种平台上轻松部署，并减少环境配置和依赖管理的工作量。

## 特性

- 从文本中提取信息
  - 实体抽取
  - 关系抽取
  - 事件抽取
  - 观点抽取
- 从文档中提取信息
  - 实体抽取
  - 关系抽取
  - 事件抽取
  - 观点抽取


## 安装

1. 克隆仓库：
    ```bash
    git clone https://github.com/llhy60/UIE-ONNXRunTime.git
    ```
2. 进入项目目录：
    ```bash
    cd UIE-ONNXRunTime
    ```
3. 安装所需依赖：
    ```bash
    pip install -r requirements.txt
    ```

4. 模型下载：

   ```shell
   百度云链接: https://pan.baidu.com/s/1ZCmLQuA0PPm_owvCVf8Hnw 提取码: rchp
   ```

## 使用方法

1. 对文本进行信息抽取：

    ```python
    python main.py --model_config ./configs/uie.yaml --text_content "北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。" --schema '["法院","原告","被告"]' 
    ```

    > [INFO] Results: {
    >     "法院": "北京市海淀区人民法院",
    >     "原告": "张三",
    >     "被告": "B公司"
    > }

2. 对文档进行信息抽取：
    ```python
    python main.py --model_config ./configs/uie-x.yaml --image_path ./images/test.jpg --schema '["姓名", "性别", {"科目": ["成绩", "满分", "占比"]}]'
    ```

    > [INFO] Results: [
    >     {
    >         "姓名": [
    >             {
    >                 "text": "样例",
    >                 "start": 21,
    >                 "end": 23,
    >                 "probability": 0.9497581307018521,
    >                 "bbox": [
    >                     [
    >                         159,
    >                         188,
    >                         192,
    >                         206
    >                     ]
    >                 ]
    >             }
    >         ],
    >         "性别": [
    >             {
    >                 "text": "男",
    >                 "start": 40,
    >                 "end": 41,
    >                 "probability": 0.9995921071066505,
    >                 "bbox": [
    >                     [
    >                         160,
    >                         212,
    >                         177,
    >                         230
    >                     ]
    >                 ]
    >             }
    >         ],
    >         "科目": [
    >             {
    >                 "text": "外语",
    >                 "start": 165,
    >                 "end": 167,
    >                 "probability": 0.9968945392184878,
    >                 "relations": {
    >                     "成绩": [
    >                         {
    >                             "text": "80",
    >                             "start": 167,
    >                             "end": 169,
    >                             "probability": 0.5090633420284334,
    >                             "bbox": [
    >                                 [
    >                                     298,
    >                                     474,
    >                                     316,
    >                                     491
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "满分": [
    >                         {
    >                             "text": "150",
    >                             "start": 169,
    >                             "end": 172,
    >                             "probability": 0.6754068672801594,
    >                             "bbox": [
    >                                 [
    >                                     362,
    >                                     474,
    >                                     386,
    >                                     490
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "占比": [
    >                         {
    >                             "text": "53%",
    >                             "start": 172,
    >                             "end": 175,
    >                             "probability": 0.8337247801817398,
    >                             "bbox": [
    >                                 [
    >                                     431,
    >                                     474,
    >                                     455,
    >                                     490
    >                                 ]
    >                             ]
    >                         }
    >                     ]
    >                 },
    >                 "bbox": [
    >                     [
    >                         81,
    >                         472,
    >                         112,
    >                         491
    >                     ]
    >                 ]
    >             },
    >             {
    >                 "text": "数学",
    >                 "start": 155,
    >                 "end": 157,
    >                 "probability": 0.9978692108073517,
    >                 "relations": {
    >                     "成绩": [
    >                         {
    >                             "text": "78",
    >                             "start": 157,
    >                             "end": 159,
    >                             "probability": 0.6898865640897256,
    >                             "bbox": [
    >                                 [
    >                                     298,
    >                                     451,
    >                                     316,
    >                                     468
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "满分": [
    >                         {
    >                             "text": "150",
    >                             "start": 159,
    >                             "end": 162,
    >                             "probability": 0.7733284127392182,
    >                             "bbox": [
    >                                 [
    >                                     362,
    >                                     451,
    >                                     386,
    >                                     467
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "占比": [
    >                         {
    >                             "text": "52%",
    >                             "start": 162,
    >                             "end": 165,
    >                             "probability": 0.9164616863587867,
    >                             "bbox": [
    >                                 [
    >                                     431,
    >                                     451,
    >                                     456,
    >                                     467
    >                                 ]
    >                             ]
    >                         }
    >                     ]
    >                 },
    >                 "bbox": [
    >                     [
    >                         80,
    >                         447,
    >                         112,
    >                         469
    >                     ]
    >                 ]
    >             },
    >             {
    >                 "text": "综合",
    >                 "start": 175,
    >                 "end": 177,
    >                 "probability": 0.9957284089769018,
    >                 "relations": {
    >                     "成绩": [
    >                         {
    >                             "text": "182",
    >                             "start": 177,
    >                             "end": 180,
    >                             "probability": 0.6896152752318514,
    >                             "bbox": [
    >                                 [
    >                                     292,
    >                                     497,
    >                                     315,
    >                                     513
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "满分": [
    >                         {
    >                             "text": "300",
    >                             "start": 180,
    >                             "end": 183,
    >                             "probability": 0.839714950819257,
    >                             "bbox": [
    >                                 [
    >                                     361,
    >                                     497,
    >                                     386,
    >                                     513
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "占比": [
    >                         {
    >                             "text": "61%",
    >                             "start": 183,
    >                             "end": 186,
    >                             "probability": 0.953920188428711,
    >                             "bbox": [
    >                                 [
    >                                     431,
    >                                     497,
    >                                     455,
    >                                     513
    >                                 ]
    >                             ]
    >                         }
    >                     ]
    >                 },
    >                 "bbox": [
    >                     [
    >                         82,
    >                         495,
    >                         111,
    >                         514
    >                     ]
    >                 ]
    >             },
    >             {
    >                 "text": "语文",
    >                 "start": 144,
    >                 "end": 146,
    >                 "probability": 0.9974042632825331,
    >                 "relations": {
    >                     "成绩": [
    >                         {
    >                             "text": "106",
    >                             "start": 146,
    >                             "end": 149,
    >                             "probability": 0.8629147913899722,
    >                             "bbox": [
    >                                 [
    >                                     292,
    >                                     428,
    >                                     316,
    >                                     444
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "满分": [
    >                         {
    >                             "text": "150",
    >                             "start": 149,
    >                             "end": 152,
    >                             "probability": 0.9449037020990545,
    >                             "bbox": [
    >                                 [
    >                                     361,
    >                                     428,
    >                                     386,
    >                                     444
    >                                 ]
    >                             ]
    >                         }
    >                     ],
    >                     "占比": [
    >                         {
    >                             "text": "71%",
    >                             "start": 152,
    >                             "end": 155,
    >                             "probability": 0.9868335292460415,
    >                             "bbox": [
    >                                 [
    >                                     431,
    >                                     428,
    >                                     455,
    >                                     444
    >                                 ]
    >                             ]
    >                         }
    >                     ]
    >                 },
    >                 "bbox": [
    >                     [
    >                         82,
    >                         427,
    >                         112,
    >                         446
    >                     ]
    >                 ]
    >             }
    >         ]
    >     }
    > ]

3. 结果将保存在 `result` 目录中。



