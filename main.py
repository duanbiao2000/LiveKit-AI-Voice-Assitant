import asyncio

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc

load_dotenv()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit."
            "Your interface with users will be voice."
            "You should use short and concise responses, "
            "and avoiding usage of unpronouncable punctuation."
        ),
    )
    # 等待连接，并设置自动订阅为仅音频
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()

    # 创建一个语音助手对象，参数包括语音活动检测器、语音识别器、语言模型、文本转语音器和初始上下文和函数上下文
    assitant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    assitant.start(ctx.room)

    await asyncio.sleep(1)
    await assitant.say(
        "Hey, how can I help you today!", allow_interruptions=True
    )

if __name__ == "__main__":
    # 运行应用程序，传入WorkerOptions参数，其中包含entrypoint_fnc参数，值为entrypoint
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
