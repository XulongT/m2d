import torch


def compare_dicts(dict1, dict2):
    # 比较两个字典的键
    if dict1.keys() != dict2.keys():
        missing_keys = set(dict1.keys()) - set(dict2.keys())
        additional_keys = set(dict2.keys()) - set(dict1.keys())

        if missing_keys:
            print("Keys present in the first dictionary but missing in the second:", missing_keys)
        if additional_keys:
            print("Keys present in the second dictionary but missing in the first:", additional_keys)

        return False

    all_same = True
    # 逐一比较两个字典的项
    for key in dict1.keys():
        item1 = dict1[key]
        item2 = dict2[key]

        # 判断项目是否为张量
        if isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
            if not torch.allclose(item1, item2, atol=1e-7):
                print(f"Values for key '{key}' are different!")
                all_same = False
        # 如果项目是OrderedDict，则递归调用
        elif isinstance(item1, dict) and isinstance(item2, dict):
            if not compare_dicts(item1, item2):
                all_same = False
        else:
            print(f"Type mismatch for key '{key}'!")
            all_same = False

    return all_same


# 加载两个.pt文件
model_weights_1 = torch.load('experiments/sep_vqvae_root/ckpt/epoch_500.pt')
model_weights_2 = torch.load('experiments/sep_vqvae_root/ckpt/epoch_500_ori.pt')

# 检查两个模型是否完全相同
if compare_dicts(model_weights_1, model_weights_2):
    print("两个.pt文件是相同的!")
else:
    print("两个.pt文件有所不同!")
