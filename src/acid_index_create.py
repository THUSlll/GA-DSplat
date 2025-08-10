# Copyright (c) 2024 Haofei Xu
# Copyright (c) 2025 Bo Liu
# Licensed under the MIT License

import json
import pdb
import torch
import os
from tqdm import tqdm
import random

with open("/hdd/u202320081001061/acid/test/index.json", "r") as f:
    json_dict = json.load(f)
save_json = {}
for scene in tqdm(json_dict, desc="Processing scenes"):
    scene_dict = {}
    path = json_dict[scene]
    file_path = os.path.join("/hdd/u202320081001061/acid/test/", path)
    file = torch.load(file_path)
    for sc in file:
        if sc['key'] == scene:
            sample = (len(sc['cameras']) - 1) / 11
            index = 0
            context = []
            for i in range(0, 12):
                int_index = int(index)
                context.append(int_index)
                index += sample
            target = random.sample(range(len(sc['cameras'])), 3)
            scene_dict['context'] = context
            scene_dict['target'] = target
            continue
    save_json[scene] = scene_dict

# 保存为JSON文件
with open("/home/u202320081001061/GaussinFuser/depthsplat/assets/evaluation_index_acid_12v.json", "w") as f:
    json.dump(save_json, f, indent=2)

print("处理完成！结果已保存到 evaluation_index_acid_12v.json")

