import torch

from sentence_transformers import SentenceTransformer

from typing import Optional
from functools import lru_cache
from nonebot.log import logger

from .history_attn import HistoryAttention
from .template import Role
from .config import Config

config = Config()

class HistoryManager:
    def __init__(self, init_history: Optional[list[dict[str, str]]] = []) -> None:
        self.model_embed = SentenceTransformer('uer/sbert-base-chinese-nli')
        self.model_sim = HistoryAttention(768, 768)
        self.model_sim.load_state_dict(torch.load(r'../../AI/history_attention/history_attn.pth',map_location=config.default_device))
        self.model_sim.eval()
        self.history = init_history  # 存储历史对话
        self.gamma = config.gamma # 历史衰减系数

    def add_to_history(self, input_text: str, model_reply: str) -> None:
        self.history.append({"role":Role.USER.value,"content":input_text})
        self.history.append({"role":Role.ASSISTANT.value,"content":model_reply})

    def sample_history(self, input_text: str) -> list[dict[str,str]]:
        similarities = (self.gamma ** torch.arange(len(self.history) - 1, -1, -1) *
                        self._get_sim(input_text, [context["content"] for context in self.history]))
        
        logger.debug(f"Similarities sum:{torch.sum(similarities)}")
        probabilities = self._softmax(similarities)
        logger.debug(f"History probabilities:{probabilities}")
        sampled_index = torch.multinomial(probabilities, 1).item()
        for i, context in enumerate(self.history):
            logger.debug(f"context:{context}\npossibility:{probabilities[i]}\nscore:{similarities[i]}")
             
        sampled_history = []
        index = sampled_index
        sampled_history.append(self._get_role_content_pair(index))
        if self.history[sampled_index]["role"] == Role.ASSISTANT.value:
            while index-1>=0:
                index -= 1
                role = self.history[index]["role"]
                sampled_history.append(self._get_role_content_pair(index))
                if role == Role.USER.value:
                    break
            sampled_history.reverse()
            return sampled_history
        elif self.history[sampled_index]["role"] == Role.USER.value:
            while index+1<len(self.history):
                index += 1
                role = self.history[index]["role"]
                sampled_history.append(self._get_role_content_pair(index))
                if role == Role.ASSISTANT.value:
                    break
            return sampled_history
        else:
            return [self.history[sampled_index]]

    def _get_sim(self, input_text: str, history_text: list[str]) -> list:
        input_embedded = torch.Tensor(self.get_embedding(input_text))
        history_text_embedded = torch.Tensor([self.get_embedding(text) for text in history_text])
        
        #添加batch
        history_text_embedded = history_text_embedded.unsqueeze(0)
        input_embedded = input_embedded.unsqueeze(0)
        
        return self.model_sim(input_embedded, history_text_embedded).squeeze(0)

    @lru_cache(maxsize=128)
    def get_embedding(self, text: str) -> torch.Tensor:
        return self.model_embed.encode(text)

    def _softmax(self, values: list[float]) -> torch.Tensor:
        tensor = values.clone().detach()
        return torch.softmax(tensor, dim=0)

    def _get_role_content_pair(self,index:int) -> dict[str,str]:
        return {"role":self.history[index]["role"],"content":self.history[index]["content"]}