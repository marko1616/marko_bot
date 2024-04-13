from pydantic import BaseModel, Extra


class Config(BaseModel, extra=Extra.ignore):
    """Plugin Config Here"""
    default_device: str = "cuda"

    top_p: float = 0.75
    top_k: int = 10
    temperature: float = 0.60
    cutoff_len: int = 1024
    max_gen_len: int = 256
    gamma: int = 0.99

    model_lora_path: str | None = None

    system_prompt: str = ""
    template: str = "llama2_zh"
    quantization_mode: str | None = "8bit"  # in 4bit 8bit None
    load_in_8bit_fp32_cpu_offload: bool = False

    enable_dispatch: bool = True
    device_map: dict[str, int | str] | None = {"model": 0, "lm_head": 0}
    max_memory_map: dict[str | int, str] = {1: "22GiB", "cpu": "10GiB"}
    mode_loading_param: dict = {
        "low_cpu_mem_usage": True,
        "pretrained_model_name_or_path": r"J:\AI\models\markobot",
        "trust_remote_code": False
    }

    save_history_to_disk: bool = True
    history_path: str = r"./data/history.json"
