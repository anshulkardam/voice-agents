from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    ChatContext,
    get_job_context,
)
import requests, json, os, re, sys
from bs4 import BeautifulSoup
import pdfplumber
from datetime import datetime
from openai import OpenAI
from livekit.plugins import openai, silero, cartesia
from dotenv import load_dotenv

load_dotenv()


def process_link(link: str):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        # Preprocess the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    llm = OpenAI()

    job_schema = {
        "type": "object",
        "properties": {
            "job title": {"type": "string"},
            "job type": {
                "type": "string",
                "enum": ["full-time", "part-time", "contract", "internship"],
            },
            "location": {"type": "string"},
            "start date": {"type": "string"},
            "qualifications": {"type": "string"},
            "responsibilities": {"type": "string"},
            "benefits": {"type": "string"},
        },
        "required": ["job title", "job type", "qualifications", "responsibilities"],
        "additionalProperties": False,
    }

    response = llm.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": f"You are Azio. A link summarizing agent. All information you need about the job is here: {text}",
            },
            {
                "role": "user",
                "content": f"Following the given response format, summarize the relevant information about this job.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "job_schema",
                "strict": False,
                "schema": job_schema,
            }
        },
    )

    data = response.output_text

    try:
        parsed = json.loads(data)  # convert JSON string -> Python dict
        print(json.dumps(parsed, indent=2))  # pretty print
        return parsed
    except json.JSONDecodeError:
        print("Invalid JSON output:")
        return None


def parse_pdf_to_text(file_path, context_file_path=None):
    """
    Parse a PDF file into plain text, removing bulletpoints and special signs, but preserving characters like @ and .

    Args:
        file_path (str): Path to the PDF file.
        context_file_path (str, optional): Path to the JSON context file. Defaults to None.

    Returns:
        str: The parsed text.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

            # Remove bulletpoints and special signs, but preserve characters like @ and .
            text = re.sub(r"[\n\t\r]", " ", text)
            text = re.sub(r"[^\w\s\.,!?@:\-]", "", text)
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            if context_file_path:
                with open(context_file_path, "r") as f:
                    context_data = json.load(f)
                    # You can now use the context data as needed
                    print("Context Data:")
                    print(json.dumps(context_data, indent=4))

            return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None


def process_pdf(pdf_path):

    try:
        text = parse_pdf_to_text(pdf_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    client = OpenAI()

    resume_schema = {
        "type": "object",
        "properties": {
            "education": {"type": "string"},
            "skills": {"type": "string"},
            "languages": {"type": "string"},
            "job experience": {"type": "string"},
            "publications": {"type": "string"},
            "location": {"type": "string"},
            "phone number": {"type": "integer"},
            "linkedin": {"type": "string"},
            "github": {"type": "string"},
            "google scholar": {"type": "string"},
        },
        "required": ["education", "skills", "job experience"],
        "additionalProperties": False,
    }

    completion = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": f"You are a resume summarizing aganet. All information you need about the candidate is here: {text}",
            },
            {
                "role": "user",
                "content": f"Following the given response format, summarize the relevant information about this candidate.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "resume_schema",
                "strict": False,
                "schema": resume_schema,
            }
        },
    )
    # Parse the JSON response
    candidate_data = json.loads(completion.output_text)

    print(json.dumps(candidate_data, indent=2))

    return candidate_data


class Assistant(Agent):
    def __init__(
        self, chat_ctx: ChatContext, job_context: dict, candidate_context: dict
    ) -> None:
        # Format the context information for the instructions
        print(f"job_context incoming: {job_context}")
        print(f"candidate_context info: {candidate_context}")

        job_info = (
            json.dumps(job_context, indent=2)
            if job_context
            else "No job information provided"
        )
        resume_info = (
            json.dumps(candidate_context, indent=2)
            if candidate_context
            else "No resume information provided"
        )

        print(f"job info: {job_info}")
        print(f"resume info: {resume_info}")

        today = datetime.now().strftime("%B %d, %Y")

        # Pass the pre-populated chat context to the parent Agent
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=f"""You are a voice assistant conducting a final round of interview.

            TODAY'S DATE: {today}

            JOB INFORMATION:
            The candidate is interviewing for the following position:
            {job_info}

            CANDIDATE'S RESUME:
            {resume_info}

            Your role:
            1. Ask thoughtful interview questions related to the specific job requirements and responsibilities mentioned above
            2. Reference the candidate's experience and skills from their resume when asking questions
            3. Listen carefully to the candidate's responses
            4. Provide brief, constructive feedback after each answer (1-2 sentences max)
            5. Ask follow-up questions when appropriate based on their resume and the job requirements
            6. Keep the conversation natural and professional
            7. Reject them if they dont have an answer and end the interview

            When the user wants to end the interview (says "bye", "goodbye", "end interview", "that's all", etc.):
            - Thank them for their time
            - Wish them good luck
            - Use the end_interview tool to gracefully end the session

            Keep your feedback concise and straight forward. Be Strict to the candidate.""",
        )

    @function_tool()
    async def end_interview(self):
        """Call this function when the user wants to end the interview or says goodbye."""
        # Get job context and shutdown gracefully
        job_ctx = get_job_context()
        job_ctx.shutdown(reason="Interview completed by user request")
        return "Interview ended successfully"


async def entrypoint(ctx: JobContext):
    try:

        pdf = "/home/karzemo/repos/self/interview_agent/AnshulKardam.pdf"
        link = "https://wellfound.com/jobs/3276104-artificial-intelligence-engineer"

        print("processing context \n\n")
        # Process the contents and extract useful information
        JOB_CONTEXT = process_link(link)
        CANDIDATE_CONTEXT = process_pdf(pdf)

        print("processed context! \n\n", JOB_CONTEXT, CANDIDATE_CONTEXT)

        await ctx.connect()

        print("connected! \n\n")

        chat_ctx = ChatContext()

        session = AgentSession(
            vad=silero.VAD.load(),
            stt=cartesia.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
        )

        await session.start(
            agent=Assistant(
                chat_ctx=chat_ctx,
                job_context=JOB_CONTEXT,
                candidate_context=CANDIDATE_CONTEXT,
            ),
            room=ctx.room,
        )

        # Initial prompt from assistant
        assistant_msg = await session.generate_reply(
            instructions="In one sentence tell the user that you will conduct a final interview to either select or reject them. No filler or explanation. Then pause."
        )
        chat_ctx.add_message(role="assistant", content=assistant_msg)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
