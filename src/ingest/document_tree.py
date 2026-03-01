import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from .paragraphs import (
    SILICONFLOW_ENDPOINT,
    SILICONFLOW_DEFAULT_MODEL,
    REQUIRED_API_KEY_ENV_NAMES,
)


def _get_llm_config() -> dict:
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_API_KEY")
    if not api_key:
        for key_name in REQUIRED_API_KEY_ENV_NAMES:
            api_key = os.environ.get(key_name)
            if api_key:
                break
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT).strip()
    model = os.environ.get("SILICONFLOW_READING_MODEL", SILICONFLOW_DEFAULT_MODEL)
    return {"api_key": api_key, "endpoint": endpoint, "model": model}


def preflight_tree_builder() -> dict:
    config = _get_llm_config()
    if not config["api_key"]:
        return {"ready": False, "reason": "no_api_key"}
    if not config["endpoint"]:
        return {"ready": False, "reason": "no_endpoint"}
    if not config["endpoint"].startswith("https://"):
        return {"ready": False, "reason": "endpoint_not_https"}
    return {"ready": True, **config}


@dataclass
class TreeNode:
    title: str
    start_page: int
    end_page: int
    level: int = 1
    node_id: str = ""
    summary: str = ""
    children: list["TreeNode"] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        result = {
            "title": self.title,
            "start_index": self.start_page,
            "end_index": self.end_page,
            "level": self.level,
        }
        if self.node_id:
            result["node_id"] = self.node_id
        if self.summary:
            result["summary"] = self.summary
        if self.children:
            result["nodes"] = [c.to_dict() for c in self.children]
        return result


def call_llm(prompt: str, config: dict, max_tokens: int = 4000) -> tuple[str, dict]:
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
    
    meta = {"model": config["model"], "endpoint": config["endpoint"], "success": False, "error": None}
    
    try:
        req = urllib.request.Request(
            config["endpoint"],
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            meta["success"] = True
            meta["prompt_tokens"] = data.get("usage", {}).get("prompt_tokens", 0)
            meta["completion_tokens"] = data.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        content = f"Error: {e}"
        meta["error"] = str(e)
    
    return content, meta


def build_toc_extraction_prompt(page_texts: dict[int, str], max_pages: int = 15) -> str:
    pages_sample = []
    for page_num in sorted(page_texts.keys())[:max_pages]:
        text = page_texts[page_num][:2000]
        pages_sample.append(f"--- Page {page_num} ---\n{text}")
    
    prompt = f"""You are an expert at analyzing academic paper structures.

Given the first {len(pages_sample)} pages of a paper, extract the table of contents / section hierarchy.

Rules:
1. Identify main sections (level 1) and subsections (level 2+)
2. Estimate the starting page for each section based on content
3. Return ONLY valid JSON array
4. Use "Introduction" as first section if paper starts with introduction

Return this EXACT format:
[
  {{"title": "Section Title", "level": 1, "start_page": 1}},
  {{"title": "Subsection Title", "level": 2, "start_page": 5}},
  ...
]

Pages:
{chr(10).join(pages_sample)}

JSON:"""
    return prompt


def parse_sections(content: str) -> list[dict]:
    sections = []
    try:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        sections = json.loads(content)
    except json.JSONDecodeError:
        import re
        matches = re.findall(r'\{[^{}]*"title"[^{}]*"level"[^{}]*\}', content)
        for m in matches:
            try:
                sections.append(json.loads(m))
            except:
                pass
    return sections if isinstance(sections, list) else []


def build_tree_from_sections(sections: list[dict], max_page: int) -> list[TreeNode]:
    if not sections:
        return []
    
    nodes = []
    stack = []
    
    for sec in sections:
        level = sec.get("level", 1)
        title = sec.get("title", "Unknown")
        start_page = min(sec.get("start_page", 1), max_page)
        
        node = TreeNode(
            title=title,
            start_page=start_page,
            end_page=start_page,
            level=level,
        )
        
        while stack and stack[-1].level >= level:
            stack.pop()
        
        if not stack:
            nodes.append(node)
        else:
            stack[-1].children.append(node)
            stack[-1].end_page = start_page
        
        stack.append(node)
    
    for node in nodes:
        _propagate_end_pages(node, max_page)
    
    return nodes


def _propagate_end_pages(node: TreeNode, max_page: int):
    if node.children:
        for child in node.children:
            _propagate_end_pages(child, max_page)
        node.end_page = node.children[-1].end_page
    else:
        node.end_page = max_page


def extract_page_texts(run_dir: Path) -> dict[int, str]:
    pages_dir = run_dir / "pages"
    page_texts = {}
    
    if pages_dir.exists():
        for page_file in sorted(pages_dir.glob("*.txt")):
            try:
                page_num = int(page_file.stem.replace("page_", ""))
                with open(page_file, "r", encoding="utf-8") as f:
                    page_texts[page_num] = f.read()
            except (ValueError, OSError):
                continue
    
    if not page_texts:
        blocks_file = run_dir / "text" / "blocks_norm.jsonl"
        if blocks_file.exists():
            with open(blocks_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        block = json.loads(line)
                        page = block.get("page", 1)
                        text = block.get("text", "")
                        if page in page_texts:
                            page_texts[page] += "\n" + text
                        else:
                            page_texts[page] = text
                    except:
                        continue
    
    return page_texts


def build_section_summary_prompt(section_text: str) -> str:
    prompt = f"""Summarize this paper section in 1-2 sentences for retrieval purposes:

{section_text[:3000]}

Summary:"""
    return prompt


def add_summaries_to_tree(nodes: list[TreeNode], page_texts: dict[int, str], config: dict):
    for node in nodes:
        section_text = ""
        for page_num in range(node.start_page, node.end_page + 1):
            if page_num in page_texts:
                section_text += page_texts[page_num][:1000] + " "
        
        if section_text:
            prompt = build_section_summary_prompt(section_text)
            content, _ = call_llm(prompt, config, max_tokens=500)
            if content and not content.startswith("Error"):
                node.summary = content.strip()[:300]
        
        if node.children:
            add_summaries_to_tree(node.children, page_texts, config)


def verify_tree_structure(nodes: list[TreeNode], page_texts: dict[int, str], config: dict) -> dict:
    stats = {"verified": 0, "adjusted": 0, "total": 0}
    
    for node in nodes:
        stats["total"] += 1
        
        if node.start_page in page_texts:
            page_text = page_texts[node.start_page][:1500]
            
            prompt = f"""Check if section "{node.title}" actually starts on page {node.start_page}.

Page {node.start_page}: {page_text[:500]}

Does it start here? Reply JSON: {{"starts_here": true/false, "actual_page": null}}

JSON:"""
            
            content, _ = call_llm(prompt, config, max_tokens=200)
            sections = parse_sections(content)
            
            if sections and isinstance(sections, list) and len(sections) > 0:
                first = sections[0]
                if first.get("starts_here", True):
                    stats["verified"] += 1
                elif first.get("actual_page") and first["actual_page"] != node.start_page:
                    node.start_page = first["actual_page"]
                    stats["adjusted"] += 1
        
        if node.children:
            child_stats = verify_tree_structure(node.children, page_texts, config)
            stats["verified"] += child_stats["verified"]
            stats["adjusted"] += child_stats["adjusted"]
            stats["total"] += child_stats["total"]
    
    return stats


def build_document_tree(run_dir: Path, add_summaries: bool = True, add_verification: bool = True) -> dict:
    page_texts = extract_page_texts(run_dir)
    
    if not page_texts:
        return {"error": "no_page_texts", "tree": []}
    
    preflight = preflight_tree_builder()
    if not preflight.get("ready"):
        return {"error": preflight.get("reason", "preflight_failed"), "tree": []}
    
    config = {"api_key": preflight["api_key"], "endpoint": preflight["endpoint"], "model": preflight["model"]}
    
    max_page = max(page_texts.keys())
    
    prompt = build_toc_extraction_prompt(page_texts, max_pages=min(20, max_page))
    content, meta = call_llm(prompt, config)
    
    sections = parse_sections(content)
    
    if not sections:
        return {"error": "no_sections_extracted", "tree": [], "meta": meta}
    
    nodes = build_tree_from_sections(sections, max_page)
    
    if add_verification and len(nodes) > 1:
        verify_stats = verify_tree_structure(nodes, page_texts, config)
        meta["verification"] = verify_stats
    
    if add_summaries:
        add_summaries_to_tree(nodes, page_texts, config)
    
    tree_dict = [n.to_dict() for n in nodes]
    
    return {
        "tree": tree_dict,
        "meta": meta,
        "total_pages": max_page,
        "total_sections": len(sections),
    }


def run_document_tree(run_dir: Path, manifest: Any, qa_dir: Path) -> dict:
    result = build_document_tree(
        run_dir=run_dir,
        add_summaries=True,
        add_verification=True,
    )
    
    tree_path = run_dir / "document_tree.json"
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump({"structure": result.get("tree", [])}, f, indent=2, ensure_ascii=False)
    
    return result
