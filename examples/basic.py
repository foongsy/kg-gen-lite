from dotenv import load_dotenv
from kg_gen.models import Graph  # noqa: F401
from kg_gen import KGGen
import json  # noqa: F401
import os  # noqa: F401

load_dotenv()

MODEL = os.getenv(
    "LLM_MODEL",
    "vercel_ai_gateway/google/gemini-3.5-flash",
)
API_KEY = os.getenv("LLM_API_KEY")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "minimal") or None


def _env_ssl_verify() -> bool:
    return os.getenv("LLM_SSL_VERIFY", "true").lower() not in ("false", "0", "no")


text = """
A Place for Demons
IT WAS FELLING NIGHT, and the usual crowd had gathered at the
Waystone Inn. Five wasn’t much of a crowd, but five was as many as the
Waystone ever saw these days, times being what they were.
Old Cob was filling his role as storyteller and advice dispensary. The
men at the bar sipped their drinks and listened. In the back room a young
innkeeper stood out of sight behind the door, smiling as he listened to the
details of a familiar story.
“When he awoke, Taborlin the Great found himself locked in a high
tower. They had taken his sword and stripped him of his tools: key, coin,
and candle were all gone. But that weren’t even the worst of it, you see…”
Cob paused for effect, “…cause the lamps on the wall were burning blue!”
Graham, Jake, and Shep nodded to themselves. The three friends had
grown up together, listening to Cob’s stories and ignoring his advice.
Cob peered closely at the newer, more attentive member of his small
audience, the smith’s prentice. “Do you know what that meant, boy?”
Everyone called the smith’s prentice “boy” despite the fact that he was a
hand taller than anyone there. Small towns being what they are, he would
most likely remain “boy” until his beard filled out or he bloodied someone’s
nose over the matter.
"""

kg = KGGen(ssl_verify=_env_ssl_verify())
# with open("tests/data/kingkiller_chapter_one.txt", "r", encoding="utf-8") as f:
#     text = f.read()

graph = kg.generate(
    input_data=text,
    model=MODEL,
    api_key=API_KEY,
    temperature=TEMPERATURE,
    reasoning_effort=REASONING_EFFORT,
    context="Kingkiller Chronicles",
    output_folder="./examples/",
    no_dspy=True,
)
# with open("./examples/graph.json", "r") as f:
#     graph = Graph(**json.load(f))

KGGen.visualize(graph, "./examples/basic-graph.html", True)
