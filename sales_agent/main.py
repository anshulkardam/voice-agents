from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero, cartesia
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_context():
    """Load all files from context directory"""
    context_dir = Path("context")
    context_dir.mkdir(exist_ok=True)

    all_content = ""
    for file_path in context_dir.glob("*"):
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
                all_content += f"\n=== {file_path.name} ===\n{content}\n"
            except:
                pass

    return all_content.strip() or "No files found"


class TechnicalAgent(Agent):
    """Technical specialist for detailed product specifications"""

    def __init__(self):
        SALES_CONTEXT = load_context()

        llm = openai.LLM(model="gpt-4o-mini")
        stt = cartesia.STT()
        tts = cartesia.TTS(voice="bf0a246a-8642-498a-9950-80c35e9276b5")
        vad = silero.VAD.load()

        instructions = f"""
        You are a technical specialist communicating by voice. All text that you return
        will be spoken aloud, so don't use things like bullets, slashes, or any
        other non-pronouncable punctuation.

        You specialize in technical details, specifications, and implementation questions.
        Focus on technical accuracy and depth.

        You have access to the following company information:

        {SALES_CONTEXT}

        CRITICAL RULES:
        - ONLY use information from the context above
        - Focus on technical specifications and features
        - Explain technical concepts clearly for non-technical users
        - DO NOT make up technical details

        You can transfer to other specialists:
        - Use switch_to_sales() to return to general sales
        - Use switch_to_pricing() for pricing questions
        """

        super().__init__(instructions=instructions, stt=stt, llm=llm, tts=tts, vad=vad)

    async def on_enter(self):
        """Called when entering this agent"""
        print("Current Agent: 💻 Technical Specialist 💻")
        await self.session.say(
            "Hi, I'm the technical specialist. I can help you with detailed technical questions about our products."
        )

    @function_tool
    async def switch_to_sales(self):
        """Switch to a sales representative"""
        await self.session.generate_reply(
            user_input="Confirm you are transferring to the sales team"
        )
        return SalesAgent()

    @function_tool
    async def switch_to_pricing(self):
        """Switch to pricing specialist"""
        await self.session.generate_reply(
            user_input="Confirm you are transferring to a pricing specialist"
        )
        return PricingAgent()


class PricingAgent(Agent):
    """Pricing specialist for budget and cost discussions"""

    def __init__(self):
        SALES_CONTEXT = load_context()

        llm = openai.LLM(model="gpt-4o-mini")
        stt = cartesia.STT()
        tts = cartesia.TTS(voice="4df027cb-2920-4a1f-8c34-f21529d5c3fe")
        vad = silero.VAD.load()

        instructions = f"""
        You are a pricing specialist communicating by voice. All text that you return
        will be spoken aloud, so don't use things like bullets, slashes, or any
        other non-pronouncable punctuation.

        You specialize in pricing, budgets, discounts, and financial aspects.
        Help customers find the best value for their needs.

        You have access to the following company information:

        {SALES_CONTEXT}

        CRITICAL RULES:
        - ONLY use pricing information from the context above
        - Focus on value proposition and ROI
        - Help customers understand pricing tiers and options
        - DO NOT make up prices or discounts

        You can transfer to other specialists:
        - Use switch_to_sales() to return to general sales
        - Use switch_to_technical() for technical questions
        """

        super().__init__(instructions=instructions, stt=stt, llm=llm, tts=tts, vad=vad)

    async def on_enter(self):
        """Called when entering this agent"""
        print("Current Agent: 💰 Pricing Agent 💰")
        await self.session.say(
            "Hello, I'm the pricing specialist. I can help you understand our pricing options and find the best value for your needs."
        )

    @function_tool
    async def switch_to_sales(self):
        """Switch back to sales representative"""
        await self.session.generate_reply(
            user_input="Confirm you are transferring to the sales team"
        )
        return SalesAgent()

    @function_tool
    async def switch_to_technical(self):
        """Switch to technical specialist"""
        await self.session.generate_reply(
            user_input="Confirm you are transferring to technical support"
        )
        return TechnicalAgent()


class SalesAgent(Agent):
    def __init__(self):
        # Load context once at startup
        context = load_context()
        print(f"📄 Loaded context: {len(context)} characters")

        llm = openai.LLM(model="gpt-4o-mini")
        stt = cartesia.STT(language="hi")
        tts = cartesia.TTS(voice="faf0731e-dfb9-4cfc-8119-259a79b27e12")
        vad = silero.VAD.load()

        # Put ALL context in system instructions
        instructions = f"""
        You are a sales agent communicating by voice in Hindi or Hinglish. All text that you return
        will be spoken aloud, so don't use things like bullets, slashes, or any
        other non-pronouncable punctuation. Do not use English.

        You have access to the following company information:

        {context}

        CRITICAL RULES:
        - You must always respond fully in Hindi, not English transliteration.
        - ONLY use information from the context above
        - If asked about something not in the context, say "Maaf kijiye, is baare mai mai aapki is time madat nahi karpaungi"
        - DO NOT make up prices, features, or any other details
        - Quote directly from the context when possible
        - Be a sales agent but only use the provided information
        - If a user asks for more info, transfer to pricing specialist
        """

        super().__init__(instructions=instructions, stt=stt, llm=llm, tts=tts, vad=vad)

    # This tells the Agent to greet the user as soon as they join, with some context about the greeting.
    async def on_enter(self):
        self.session.generate_reply(
            user_input="Give a short and formal 1 sentence greeting in Hindi starting with 'Namaste!'. Offer to answer any questions."
        )

    @function_tool
    async def switch_to_tech_support(self):
        """Switch to a technical support rep"""
        await self.session.generate_reply(
            user_input="Confirm you are transferring to technical support"
        )
        return TechnicalAgent()

    @function_tool
    async def switch_to_pricing(self):
        """Switch to pricing specialist"""
        await self.session.generate_reply(
            user_input="Confirm you are transferring to a pricing specialist"
        )
        return PricingAgent()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = SalesAgent()
    session = AgentSession()
    await session.start(room=ctx.room, agent=agent)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
