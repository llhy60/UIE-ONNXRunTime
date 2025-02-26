import os
import json
import argparse
import time
from pipelines.uie_predictor import UIEPredictor
from pipelines.uiex_predictor import UIEXPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='./configs/uie.yaml')
    parser.add_argument('--text_content', type=str, default="北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。")
    parser.add_argument('--schema', type=str, default='["法院","原告","被告"]')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./result')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    schema = args.schema
    model_config = args.model_config
    text_content = args.text_content
    image_path = args.image_path
    results = {}
    uie_res = {}

    if schema is not None:
        schema = json.loads(schema)
    else:
        schema = []

    start_time = time.time()

    if image_path is not None:
        input_data = {"doc": image_path}
        uiex_model = UIEXPredictor(model_config, schema)
        uie_res = uiex_model(input_data)
        results = uie_res
    else:
        uie_model = UIEPredictor(model_config, schema)
        uie_res = uie_model(text_content)[0]
        for schema_key in schema:
            if schema_key not in uie_res.keys():
                results[schema_key] = ""
            else:
                uie_res[schema_key] = sorted(
                    uie_res[schema_key], key=lambda x: x["start"]
                )
                results[schema_key] = ",".join(
                    [
                        uie_res[schema_key][i]["text"]
                        for i in range(len(uie_res[schema_key]))
                    ]
                )
    results = json.dumps(results, ensure_ascii=False, indent=4)
    end_time = time.time()
    print(f"[INFO] Results: {results}")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, "res.json"), "w") as f:
        f.write(results)
    print(f"[INFO] Results saved in {args.save_dir}")
    print(f"[INFO] Time cost: {(end_time - start_time):.4f} s")