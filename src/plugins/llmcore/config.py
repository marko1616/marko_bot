from pydantic import BaseModel, Extra


class Config(BaseModel, extra=Extra.ignore):
    """Plugin Config Here"""
    default_device:str="cuda"
    
    top_p:float=0.75
    top_k:int=10
    temperature:float=0.60
    cutoff_len:int=1024
    max_gen_len:int=256
    
    gamma:int = 0.99

    model_path:str=r"../../AI/models/Qwen1.5-14B-Chat"
    model_lora_path:str|None=None
    system_prompt:str=""
    template:str="qwen"
    quantization_mode:str|None="8bit"#in 4bit 8bit None
    
    enable_dispatch:bool = True
    #device_map:dict[str,int|str]|None={'model.embed_tokens': 'cpu', 'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 'cpu', 'model.layers.36': 'cpu', 'model.layers.37': 'cpu', 'model.layers.38': 'cpu', 'model.layers.39': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}
    device_map:dict[str,int|str]|None={'model.embed_tokens': 'cpu', 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 'cpu', 'model.layers.32': 'cpu', 'model.layers.33': 'cpu', 'model.layers.34': 'cpu', 'model.layers.35': 'cpu', 'model.layers.36': 'cpu', 'model.layers.37': 'cpu', 'model.layers.38': 'cpu', 'model.layers.39': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}
    max_memory_map:dict[str|int,str]={1: "22GiB","cpu": "10GiB"}
    low_cpu_mem_usage:bool = True

    save_history_to_disk:bool = True
    history_path:str = r"./data/history.json"