import enum
from typing import Annotated
from livekit.agents import llm
import logging

# 获取名为"temperature-control"的日志记录器
logger = logging.getLogger("temperature-control")
# 设置日志记录器的级别为INFO
logger.setLevel(logging.INFO)


class Zone(enum.Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"


class AssistantFnc(llm.FunctionContext):
    def __init__(self) -> None:
        super().__init__()

        self._temperature = {
            Zone.LIVING_ROOM: 22,
            Zone.BEDROOM: 20,
            Zone.KITCHEN: 24,
            Zone.BATHROOM: 23,
            Zone.OFFICE: 21,
        }

    @llm.ai_callable(description="get the temperature in a specific room")
    # `zone` 参数被注解为 `Annotated[Zone, llm.TypeInfo(description=
    # "The specific zone")]`，这意味着它的类型是 `Zone` 枚举，并且
    # 附带了描述信息，帮助用户理解该参数的用途。
    def get_temperature(
        self,
        zone: Annotated[Zone,
                        llm.TypeInfo(description="The specific zone")]
    ):
        logger.info("get temp - zone %s", zone)
        # 获取指定区域的温度
        temp = self._temperature[Zone(zone)]
        return f"The temperature in the {zone} is {temp}C"

    @llm.ai_callable(description="set the temperature in a specific room")
    def set_temperature(
        self,
        zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")],
        temp: Annotated[int,
                        llm.TypeInfo(description="The temperature to set")],
    ):
        logger.info("set temo - zone %s, temp: %s", zone, temp)
        # 将温度值赋给指定区域的温度属性
        self._temperature[Zone(zone)] = temp
        return f"The temperature in the {zone} is now {temp}C"
