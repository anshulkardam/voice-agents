import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional, List

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import deepgram, openai, silero
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

logger = logging.getLogger("restaurant-example")
logger.setLevel(logging.INFO)
load_dotenv()

# check if storage already exists
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "query-engine-storage"
if not PERSIST_DIR.exists():
    # load the documents and create the index
    documents = SimpleDirectoryReader(THIS_DIR / "data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


@function_tool
async def query_info(query: str) -> str:
    """Get more information about a specific topic"""
    query_engine = index.as_query_engine(use_async=True)
    res = await query_engine.aquery(query)
    print("Query result:", res)
    return str(res)


@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None

    reservation_time: Optional[str] = None

    order: Optional[list[str]] = None

    customer_credit_card: Optional[str] = None
    customer_credit_card_expiry: Optional[str] = None
    customer_credit_card_cvv: Optional[str] = None

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": (
                {
                    "number": self.customer_credit_card or "unknown",
                    "expiry": self.customer_credit_card_expiry or "unknown",
                    "cvv": self.customer_credit_card_cvv or "unknown",
                }
                if self.customer_credit_card
                else None
            ),
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        # summarize in yaml performs better than json
        return yaml.dump(data)


RunContext_T = RunContext[UserData]


# common functions
@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"


@function_tool()
async def get_order_details(
    context: RunContext_T,
) -> str:
    """Return the customer's current order details."""
    userdata = context.userdata
    if not userdata.order:
        return "No order has been placed yet."
    order_list = "\n".join(f"- {item}" for item in userdata.order)
    return f"Current order details for {userdata.customer_name or 'unknown'}:\n{order_list}"


@function_tool()
async def update_order(
    items: Annotated[List[str], Field(description="List of items to add to the order")],
    context: RunContext_T,
) -> str:
    """Add items to the customer's order."""
    userdata = context.userdata
    if userdata.order is None:
        userdata.order = []
    userdata.order.extend(items)
    order_list = ", ".join(userdata.order)
    return f"Order updated for {userdata.customer_name or 'unknown'}. Current order: {order_list}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [
                item for item in truncated_chat_ctx.items if item.id not in existing_ids
            ]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as assistant message
        chat_ctx.add_message(
            role="system",  # role=system works for OpenAI's LLM and Realtime API
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(
        self, name: str, context: RunContext_T
    ) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."


class Greeter(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist.\n"
                "you can get the restaurant details by calling query_info tool"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent using tools."
            ),
            llm=openai.LLM(parallel_tool_calls=False),
            tools=[query_info, update_order, get_order_details],
            tts=deepgram.TTS(),
        )

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation.
        This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation time,
        customer name and phone number."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a takeaway order.
        This includes handling orders for pickup, delivery, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("takeaway", context)


class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a reservation agent at a restaurant. Your jobs are to ask for "
            "the reservation time, then customer's name, and phone number. "
            "check if the reservation time and reservation date is available for the restauraunt from query_info tool"
            "if time not available, suggest a good alternative"
            "then confirm the reservation details with the customer.",
            tools=[
                update_name,
                update_phone,
                to_greeter,
                query_info,
                update_order,
                get_order_details,
            ],
            tts=deepgram.TTS(),
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their reservation time.
        Confirm the time with the user before calling the function."""
        userdata = context.userdata
        userdata.reservation_time = time
        return f"The reservation time is updated to {time}"

    @function_tool()
    async def confirm_reservation(
        self, context: RunContext_T
    ) -> str | tuple[Agent, str]:
        """Called when the user confirms the reservation."""
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."

        if not userdata.reservation_time:
            return "Please provide reservation time first."

        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"Your are a takeaway agent that takes orders from the customer. "
                f"you can get the menu from query_info tool \n"
                "Clarify special requests and confirm the order with the customer."
            ),
            tools=[to_greeter, query_info, get_order_details, update_order],
            tts=deepgram.TTS(),
        )

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the order."""
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."

        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"You are a checkout agent at a restaurant. you can get the menu by calling query_info tool \n"
                "Your are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV step by step."
                "finish by saying the details will be send to your phone number shortly"
            ),
            tools=[
                update_name,
                update_phone,
                to_greeter,
                get_order_details,
                update_order,
            ],
            tts=deepgram.TTS(),
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user confirms the expense."""
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV.
        Confirm the spelling with the user before calling the function."""
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is updated to {number}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout."""
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."

        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."

        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("takeaway", context)


async def entrypoint(ctx: JobContext):

    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(),
            "reservation": Reservation(),
            "takeaway": Takeaway(),
            "checkout": Checkout(),
        }
    )
    session = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=deepgram.TTS(),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(video_enabled=False),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
