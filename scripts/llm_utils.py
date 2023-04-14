import openai
from config import Config
from llama_cpp import Llama

cfg = Config()
llm = Llama(model_path="/home/me/text-generation-webui/models/TheBloke_vicuna-7B-1.1-GPTQ-4bit-128g-GGML/ggml-vicuna-7B-q4.bin", n_ctx=20000, embedding=True, n_threads=14)


def create_chat_completion(messages, model=None, temperature=0.36, max_tokens=None)->str:
    raw_content = "\n".join(m["content"] for m in messages if m["role"]=='system')
    content = (
f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{raw_content}
Response MUST end with '</s>'  which is verbatim
### Response:
{{''')


    print("Message Content:\n", content)
    response = llm(content, stop=["</s>"], echo=False, temperature=temperature, max_tokens=max_tokens, repeat_penalty=1.2, top_k=50, top_p=0.95)
    response_text = "{" + response["choices"][0]["text"]

    print("Response:\n", response_text)
    return response_text
