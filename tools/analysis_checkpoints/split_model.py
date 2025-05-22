# 分解模型权重并保存，用于预训练模型
import os
import torch
from collections import OrderedDict


def save_split_model(model,save_key,save_dir):
    assert os.path.exists(save_dir),f"{save_dir} doesn't exist"
    save_dir=os.path.join(save_dir,'split_model')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for key in save_key:
        if key in model.keys():
            save_path=os.path.join(save_dir,f'{key}.pth')
            torch.save(model[key],save_path)


def list_children_key(model,parent_keys):
    model_state_dict=model['state_dict']
    module_keys=model_state_dict.keys()
    children_keys_dict={}
    for parent_key in parent_keys:
        children_keys=[]
        parent_key_len=len(parent_key.split('.'))
        for module_key in module_keys:
            if module_key.startswith(parent_key):
                children_key=module_key.split('.')[parent_key_len]
                if children_key not in children_keys:
                    children_keys.append(children_key)
        children_keys_dict[parent_key]=children_keys
    return children_keys_dict

def pick_specified_weights(model,keys):
    model_state_dict=model['state_dict']
    module_keys=model_state_dict.keys()
    keep_modules=dict()
    for key in keys:
        module_dict=OrderedDict()
        for module_key in module_keys:
            if module_key.startswith(key):
                child_module_key=module_key[len(key)+1:]
                module_dict[child_module_key]=model_state_dict[module_key]
        keep_modules[key]=module_dict       
    return keep_modules        

     


if __name__=="__main__":
    
    model_file=torch.load('pretrained_models/bevdet-r50-cbgs.pth',map_location=torch.device('cpu'))
    model_name='bevdet-r50-cbgs'
    if not os.path.exists(f'checkpoints/pretrained_weights/{model_name}'):
        os.mkdir(f'checkpoints/pretrained_weights/{model_name}')
    model_keys=['img_backbone', 'img_neck']
    save_modules=pick_specified_weights(model_file,model_keys)
    for name,weights in save_modules.items():
        save_name=name.split('.')[-1]
        torch.save(weights,f'checkpoints/pretrained_weights/{model_name}/{save_name}.pth')