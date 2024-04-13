import os, json

from nonebot import get_driver, on_command
from nonebot.adapters import Event
from nonebot.plugin import PluginMetadata
from nonebot.adapters import Message
from nonebot.params import CommandArg
from .utils import ChatAgent
from .config import Config

__plugin_meta__ = PluginMetadata(
    name="llmcore",
    description="",
    usage="",
    config=Config,
)

global_config = get_driver().config
driver = get_driver()
config = Config()

init_history = None
if os.path.exists(config.history_path):
    with open(config.history_path,'r') as file:
        init_history = json.load(file)
        
chat_agent = ChatAgent(config, init_history)

clear_his = on_command("清除聊天历史", aliases={"clh"}, block=True, priority=1)
@clear_his.handle()
async def _():
    chat_agent.clear_his()
    await clear_his.send("记忆清除成功捏")

stpp = on_command("设置top_p", aliases={"stpp"}, block=True, priority=1)
@stpp.handle()
async def _(args: Message = CommandArg()):
    try:
        top_p = float(args.extract_plain_text())
        assert top_p >= 0 and top_p <= 1
    except BaseException:
        await stpp.send("要输入真确的top_p哦(0-1)float")
        return
    chat_agent.top_p = top_p
    await stpp.send(f"top_p以被设定为:{top_p}呐~")

st = on_command("设置temperature", aliases={"st"}, block=True, priority=1)
@st.handle()
async def _(args: Message = CommandArg()):
    try:
        temperature = float(args.extract_plain_text())
        assert temperature >= 0 and temperature <= 1
    except BaseException:
        await st.send("要输入真确的temperature哦(0-1)float")
        return
    chat_agent.temperature = temperature
    await st.send(f"temperature以被设定为:{temperature}呐~")

cmd_msg = on_command("chat")
@cmd_msg.handle()
async def _(event: Event):
    text = ''.join(event.get_plaintext().split(" ")[1:])  # 去掉/chat的指令开头
    response = []
    async for token in chat_agent.async_stream_chat(text):
        response.append(token)
        if token in ["？","?","。",".","\n"] and response:
            await cmd_msg.send(response)
            response = []
    if response:
        await cmd_msg.send(response)

@driver.on_shutdown
async def _():
    #保存对话历史
    if config.save_history_to_disk:
        with open(config.history_path,'w') as file:
            json.dump(chat_agent.history.history,file)