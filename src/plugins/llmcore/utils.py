import torch
import asyncio
from transformers import BitsAndBytesConfig

from nonebot.log import logger

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Union, Optional, Iterator
from accelerate import infer_auto_device_map, dispatch_model

from .config import Config
from .template import templates, Role

from .history import HistoryManager

config = Config()

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
int8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=config.load_in_8bit_fp32_cpu_offload
)
quantization_configs = {
    "4bit":nf4_config,
    "8bit":int8_config,
    None:None
}

class ChatAgent():
    def __init__(self,
                 model_path: str,
                 model_lora_path: Optional[Union[None,str]]|None = None,
                 init_history: Optional[list[list[str,str]]]|None = None,
                 system_prompt: Optional[str] = config.system_prompt) -> None:
        #避免python特色的函数默认参数问题
        if init_history is None:
            init_history = []
        
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.temperature = config.temperature
        self.template = templates[config.template]
        self.cutoff_len = config.cutoff_len

        logger.info("Starting to load LLM model")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=config.trust_remote_code)
        
        assert config.quantization_mode in ["4bit","8bit",None]
        quantization_config=quantization_configs[config.quantization_mode]
        
        if config.device_map:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=config.low_cpu_mem_usage,device_map=config.device_map,quantization_config=quantization_config, trust_remote_code=config.trust_remote_code)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=config.low_cpu_mem_usage, quantization_config=quantization_config, trust_remote_code=config.trust_remote_code)
        if model_lora_path:
            if isinstance(model_lora_path,str):
                self.model = PeftModel.from_pretrained(self.model, model_lora_path)
                self.model.merge_adapter()
            elif isinstance(model_lora_path,list):
                for index, path in enumerate(model_lora_path):
                    self.model = PeftModel.from_pretrained(self.model, path, adapter_name=f"lora_{index}")
                logger.debug(self.model)
                self.model.merge_adapter()
            else:
                raise TypeError("Lora路径必须是list[str]或者str")

        if not config.device_map and config.enable_dispatch:
            device_map = infer_auto_device_map(
                self.model,
                max_memory=config.max_memory_map,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype='float16',
                verbose=True)
            # 输出层设备必须与输入层一致
            self.model = dispatch_model(self.model, device_map=device_map)

        self.model.eval()

        logger.info("LLM model loaded successfully")

        # 建立对话历史
        self.history = HistoryManager(init_history)
        self.system_prompt = system_prompt

    def build_prompt(self, text: str) -> list[int]:
        if len(self.history.history) > 0:
            context_list = self.history.sample_history(text)
            context_list.append({"role":Role.USER.value,"content":text})
        else:
            context_list = [{"role":Role.USER.value,"content":text}]
        context_list += [{"role": "assistant", "content": ""}]
    
        return self.template.encode_oneturn(self.tokenizer,context_list,self.system_prompt,self.cutoff_len)
        

    def stream_chat(self, input_text: str) -> Iterator[str]:
        logger.debug(f"temperature:{self.temperature}")
        logger.debug(f"top_p:{self.top_p}")
        input_ids = torch.IntTensor(self.build_prompt(input_text)[0]).to(config.default_device).unsqueeze(0)
        generate_input = {
            "input_ids": input_ids,
            "do_sample": True,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": 1.0,
            "max_new_tokens": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        token_list = []
        for _ in range(config.max_gen_len):
            output = self.model.generate(**generate_input)
            current_token_id = output[0][-1]
            if current_token_id == self.tokenizer.eos_token_id:
                break
            token = self.tokenizer.decode([current_token_id], skip_special_tokens=True)
            logger.debug(f"Gen token:{token}")
            token_list.append(token)
            yield token

            generate_input["input_ids"] = output[0].unsqueeze(0)
        model_reply = ''.join(token_list)
        self.history.add_to_history(input_text, model_reply)

    async def async_stream_chat(self, input_text: str):
        for token in self.stream_chat(input_text):
            yield token
            await asyncio.sleep(0)  # 让出控制权防止生成过程中event loop其他task超时

    def clear_his(self) -> None:
        self.history.history = []
