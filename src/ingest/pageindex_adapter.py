import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

VENDOR_PAGEINDEX = Path(__file__).parent.parent.parent / "vendor" / "pageindex"
sys.path.insert(0, str(VENDOR_PAGEINDEX))

SILICONFLOW_ENDPOINT = os.environ.get("SILICONFLOW_ENDPOINT") or os.environ.get("LLM_ENDPOINT_URL", "https://api.siliconflow.cn/v1/chat/completions")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_API_KEY") or os.environ.get("SILICONFLOW_TOKEN") or os.environ.get("LLM_API_KEY")
DEFAULT_MODEL = os.environ.get("SILICONFLOW_READING_MODEL", "Qwen/Qwen2.5-14B-Instruct")

if SILICONFLOW_API_KEY:
    os.environ["OPENAI_API_KEY"] = SILICONFLOW_API_KEY
os.environ["OPENAI_BASE_URL"] = SILICONFLOW_ENDPOINT

_clients = None


def get_clients():
    global _clients
    if _clients is None:
        from openai import AsyncOpenAI, OpenAI
        if not SILICONFLOW_API_KEY:
            raise ValueError("SiliconFlow API key not found.")
        _clients = {
            "sync": OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_ENDPOINT),
            "async": AsyncOpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_ENDPOINT),
        }
    return _clients


def patched_ChatGPT_API(model: str, prompt: str, api_key: str = None, chat_history: list = None) -> str:
    clients = get_clients()
    client = clients["sync"]
    for i in range(10):
        try:
            messages = (chat_history.copy() + [{"role": "user", "content": prompt}]) if chat_history else [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model=model, messages=messages, temperature=0)
            return response.choices[0].message.content
        except Exception as e:
            if i < 9:
                import time
                time.sleep(1)
    return "Error"


def patched_ChatGPT_API_with_finish_reason(model: str, prompt: str, api_key: str = None, chat_history: list = None) -> tuple:
    clients = get_clients()
    client = clients["sync"]
    for i in range(10):
        try:
            messages = (chat_history.copy() + [{"role": "user", "content": prompt}]) if chat_history else [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model=model, messages=messages, temperature=0)
            if response.choices[0].finish_reason == "length":
                return response.choices[0].message.content, "max_output_reached"
            return response.choices[0].message.content, "finished"
        except Exception as e:
            if i < 9:
                import time
                time.sleep(1)
    return "Error", "error"


import asyncio


async def patched_ChatGPT_API_async(model: str, prompt: str, api_key: str = None) -> str:
    clients = get_clients()
    client = clients["async"]
    messages = [{"role": "user", "content": prompt}]
    for i in range(10):
        try:
            response = await client.chat.completions.create(model=model, messages=messages, temperature=0)
            return response.choices[0].message.content
        except Exception as e:
            if i < 9:
                await asyncio.sleep(1)
    return "Error"


def apply_patches():
    import pageindex.page_index as pi_module
    import pageindex.utils as utils_module
    pi_module.ChatGPT_API = patched_ChatGPT_API
    pi_module.ChatGPT_API_with_finish_reason = patched_ChatGPT_API_with_finish_reason
    pi_module.ChatGPT_API_async = patched_ChatGPT_API_async
    utils_module.ChatGPT_API = patched_ChatGPT_API
    utils_module.ChatGPT_API_with_finish_reason = patched_ChatGPT_API_with_finish_reason
    utils_module.ChatGPT_API_async = patched_ChatGPT_API_async


def run_pageindex(pdf_path: str, model: str = None, toc_check_pages: int = 20, max_pages_per_node: int = 5, max_tokens_per_node: int = 8000, add_node_id: bool = True, add_node_summary: bool = True, add_doc_description: bool = False, add_node_text: bool = False) -> dict:
    from pageindex.utils import ConfigLoader
    from pageindex.page_index import page_index_main
    apply_patches()
    if model is None:
        model = DEFAULT_MODEL
    opt = ConfigLoader().load({
        "model": model,
        "toc_check_page_num": toc_check_pages,
        "max_page_num_each_node": max_pages_per_node,
        "max_token_num_each_node": max_tokens_per_node,
        "if_add_node_id": "yes" if add_node_id else "no",
        "if_add_node_summary": "yes" if add_node_summary else "no",
        "if_add_doc_description": "yes" if add_doc_description else "no",
        "if_add_node_text": "yes" if add_node_text else "no",
    })
    return page_index_main(pdf_path, opt)


def run_pageindex_simple(pdf_path: str, add_summaries: bool = True) -> dict:
    return run_pageindex(pdf_path=pdf_path, add_node_summary=add_summaries, add_node_id=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PageIndex with SiliconFlow")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    result = run_pageindex(pdf_path=args.pdf, model=args.model, add_node_summary=not args.no_summary)
    output_path = args.output
    if output_path is None:
        pdf_name = Path(args.pdf).stem
        output_path = f"results/{pdf_name}_structure.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Tree structure saved to: {output_path}")
