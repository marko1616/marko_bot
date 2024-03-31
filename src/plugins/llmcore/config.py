from pydantic import BaseModel, Extra


class Config(BaseModel, extra=Extra.ignore):
    """Plugin Config Here"""
    top_p:float=0.75
    top_k:int=10
    temperature:float=0.60
    cutoff_len:int=1024
    max_gen_len:int=256
    
    gamma:int = 0.99

    model_path:str=r"../../AI/models/markobot"
    model_lora_path:str|None=None
    system_prompt:str=""
    template:str="llama2_zh"
    
    enable_dispatch:bool = True
    device_map:dict[str,int|str]|None={'model.embed_tokens': 'cpu', 'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 'cpu', 'model.layers.36': 'cpu', 'model.layers.37': 'cpu', 'model.layers.38': 'cpu', 'model.layers.39': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}
    max_memory_map:dict[str|int,str]={1: "22GiB","cpu": "10GiB"}

    save_history_to_disk:bool = True
    history_path:str = r"./data/history.json"