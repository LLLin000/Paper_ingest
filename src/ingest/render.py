"""Obsidian renderer - generates markdown with traceability."""

import json
from pathlib import Path
from typing import Any

from .manifest import Manifest


def run_render(
    run_dir: Path,
    manifest: Manifest,
) -> tuple[int, Path]:
    """Run the render stage to generate Obsidian markdown.
    
    Returns:
        Tuple of (sections_created, output_path)
    """
    obsidian_dir = run_dir / "obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = obsidian_dir / f"{manifest.doc_id}.md"
    
    # Load all required artifacts
    paper_profile = _load_json(run_dir / "reading" / "paper_profile.json", {})
    logic_graph = _load_json(run_dir / "reading" / "logic_graph.json", {"nodes": [], "edges": []})
    themes = _load_json(run_dir / "reading" / "themes.json", {"themes": [], "cross_theme_links": []})
    synthesis = _load_json(run_dir / "reading" / "synthesis.json", {
        "executive_summary": "",
        "key_evidence_lines": [],
        "figure_table_slots": []
    })
    figure_table_index = _load_jsonl(run_dir / "figures_tables" / "figure_table_index.jsonl")
    figure_table_links = _load_json(run_dir / "figures_tables" / "figure_table_links.json", {
        "by_section": {},
        "by_fact": {},
        "by_synthesis_slot": {}
    })
    cite_map = _load_jsonl(run_dir / "citations" / "cite_map.jsonl")
    facts = _load_jsonl(run_dir / "reading" / "facts.jsonl")
    
    # Determine quality based on data availability
    quality = _determine_quality(paper_profile, synthesis, facts, cite_map)
    
    # Build the markdown content
    lines = []
    
    # Frontmatter
    lines.append("---")
    lines.append(f"doc_id: {manifest.doc_id}")
    lines.append(f"paper_type: {paper_profile.get('paper_type', 'unknown')}")
    lines.append(f"source_pdf: {manifest.input_pdf_path}")
    lines.append(f"quality: {quality}")
    lines.append("---")
    lines.append("")
    
    # Paper Profile section
    lines.append("# 论文档案 (Paper Profile)")
    lines.append("")
    lines.append(f"**论文类型 (Paper Type):** {paper_profile.get('paper_type', 'unknown')}")
    lines.append(f"**置信度 (Confidence):** {paper_profile.get('paper_type_confidence', 0.0)}")
    lines.append("")
    lines.append(f"**研究问题 (Research Problem):** {paper_profile.get('research_problem', 'N/A')}")
    lines.append("")
    lines.append(f"**声称贡献 (Claimed Contribution):** {paper_profile.get('claimed_contribution', 'N/A')}")
    lines.append("")
    lines.append(f"**阅读策略 (Reading Strategy):** {paper_profile.get('reading_strategy', 'N/A')}")
    lines.append("")
    
    # Logic Graph section
    lines.append("# 逻辑图谱 (Logic Graph)")
    lines.append("")
    nodes = logic_graph.get("nodes", [])
    edges = logic_graph.get("edges", [])
    
    if nodes:
        lines.append("## 节点 (Nodes)")
        lines.append("")
        for node in sorted(nodes, key=lambda x: x.get("id", "")):
            node_id = node.get("id", "")
            node_type = node.get("type", "unknown")
            node_label = node.get("label", "")
            lines.append(f"- **{node_id}** ({node_type}): {node_label}")
        lines.append("")
    else:
        lines.append("_无逻辑图节点可用 (No logic graph nodes available)._")
        lines.append("")
    
    if edges:
        lines.append("## 边 (Edges)")
        lines.append("")
        for edge in sorted(edges, key=lambda x: f"{x.get('from', '')}-{x.get('to', '')}"):
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            edge_type = edge.get("type", "supports")
            lines.append(f"- {from_node} --[{edge_type}]--> {to_node}")
        lines.append("")
    
    # Argument Flow
    argument_flow = logic_graph.get("argument_flow", {})
    if argument_flow.get("premises") or argument_flow.get("core_claims") or argument_flow.get("conclusions"):
        lines.append("## 论证流程 (Argument Flow)")
        lines.append("")
        if argument_flow.get("premises"):
            lines.append("### 前提 (Premises)")
            for p in argument_flow["premises"]:
                lines.append(f"- {p}")
            lines.append("")
        if argument_flow.get("core_claims"):
            lines.append("### 核心主张 (Core Claims)")
            for c in argument_flow["core_claims"]:
                lines.append(f"- {c}")
            lines.append("")
        if argument_flow.get("conclusions"):
            lines.append("### 结论 (Conclusions)")
            for c in argument_flow["conclusions"]:
                lines.append(f"- {c}")
            lines.append("")
    
    # Themes section
    lines.append("# 主题 (Themes)")
    lines.append("")
    theme_list = themes.get("themes", [])
    if theme_list:
        for theme in sorted(theme_list, key=lambda x: x.get("theme_id", "")):
            theme_id = theme.get("theme_id", "")
            theme_name = theme.get("theme_name", "Unnamed")
            theme_description = theme.get("description", "")
            lines.append(f"## {theme_id}: {theme_name}")
            lines.append("")
            lines.append(theme_description)
            lines.append("")
            
            # Related facts
            related_facts = theme.get("fact_ids", [])
            if related_facts:
                lines.append("**关联事实 (Related Facts):**")
                for fact_id in sorted(related_facts):
                    lines.append(f"- [[{fact_id}]]")
                lines.append("")
    else:
        lines.append("_无提取的主题 (No themes extracted)._")
        lines.append("")
    
    # Cross-theme links
    cross_links = themes.get("cross_theme_links", [])
    if cross_links:
        lines.append("## 跨主题链接 (Cross-Theme Links)")
        lines.append("")
        for link in sorted(cross_links, key=lambda x: f"{x.get('from', '')}-{x.get('to', '')}"):
            from_theme = link.get("from", "")
            to_theme = link.get("to", "")
            link_type = link.get("type", "related")
            lines.append(f"- {from_theme} <--> {to_theme} ({link_type})")
        lines.append("")
    
    # Evidence Index section
    lines.append("# 证据索引 (Evidence Index)")
    lines.append("")
    lines.append("| 事实ID (Fact ID) | 类别 (Category) | 陈述 (Statement) | 页码 (Page) | 边界框 (BBox) |")
    lines.append("|------------------|------------------|-------------------|-------------|---------------|")
    
    for fact in sorted(facts, key=lambda x: x.get("fact_id", "")):
        fact_id = fact.get("fact_id", "")
        category = fact.get("category", "none")
        statement = fact.get("statement", "")[:60] + ("..." if len(fact.get("statement", "")) > 60 else "")
        evidence = fact.get("evidence_pointer", {})
        page = evidence.get("page", "N/A")
        bbox = evidence.get("bbox", [])
        bbox_str = f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]" if bbox else "N/A"
        
        lines.append(f"| [[{fact_id}]] | {category} | {statement} | {page} | {bbox_str} |")
    
    lines.append("")
    
    # Key Evidence Lines from Synthesis
    lines.append("# 关键证据线 (Key Evidence Lines)")
    lines.append("")
    key_lines = synthesis.get("key_evidence_lines", [])
    if key_lines:
        for i, line in enumerate(sorted(key_lines, key=lambda x: x.get("claim", "")), 1):
            claim = line.get("claim", "")
            fact_ids = line.get("fact_ids", [])
            
            lines.append(f"## 论断 {i} (Claim {i})")
            lines.append("")
            lines.append(claim)
            lines.append("")
            
            if fact_ids:
                lines.append("**支撑事实 (Supporting Facts):**")
                for fact_id in sorted(fact_ids):
                    lines.append(f"- [[{fact_id}]]")
                lines.append("")
    else:
        lines.append("_无可用的关键证据线 (No key evidence lines available)._")
        lines.append("")
    
    # Figure/Table slots
    slots = synthesis.get("figure_table_slots", [])
    if slots:
        lines.append("# 图表占位符 (Figure/Table Slots)")
        lines.append("")
        
        for slot in sorted(slots, key=lambda x: x.get("slot_id", "")):
            slot_id = slot.get("slot_id", "")
            position = slot.get("position_hint", "unknown")
            render_mode = slot.get("render_mode", "content_only")
            asset_ids = slot.get("asset_ids", [])
            
            lines.append(f"## {slot_id}")
            lines.append("")
            lines.append(f"**位置 (Position):** {position}")
            lines.append(f"**渲染模式 (Render Mode):** {render_mode}")
            lines.append("")
            lines.append("**资源 (Assets):**")
            
            for asset_id in sorted(asset_ids):
                # Find asset info from index
                asset_info = next((a for a in figure_table_index if a.get("asset_id") == asset_id), None)
                if asset_info:
                    asset_type = asset_info.get("asset_type", "unknown")
                    page = asset_info.get("page", "N/A")
                    caption = asset_info.get("caption_text", "No caption")
                    image_path = asset_info.get("image_path", "")
                    
                    lines.append(f"- **{asset_id}** ({asset_type}) - 第{page}页 (Page {page})")
                    lines.append(f"  - 标题 (Caption): {caption}")
                    
                    if render_mode == "full_asset_embed" and image_path:
                        # Use relative path for embed
                        rel_path = Path(image_path).name
                        lines.append(f"  - ![{asset_id}](../figures_tables/assets/{rel_path})")
                    elif render_mode == "content_only":
                        lines.append(f"  - 位置 (Location): p{page}")
                        if bbox := asset_info.get("bbox_px"):
                            lines.append(f"  - 边界框(px) (BBox px): [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")
                else:
                    lines.append(f"- {asset_id} (资产信息未找到 - asset info not found)")
            
            lines.append("")
    
    # Citations section
    lines.append("# 引用 (Citations)")
    lines.append("")
    
    # Group by mapped vs unmapped
    mapped = [c for c in cite_map if c.get("mapped_ref_key")]
    unmapped = [c for c in cite_map if not c.get("mapped_ref_key")]
    
    if mapped:
        lines.append("## 已映射引用 (Mapped Citations)")
        lines.append("")
        lines.append("| 锚点ID (Anchor ID) | 映射键 (Mapped Key) | 策略 (Strategy) | 置信度 (Confidence) |")
        lines.append("|---------------------|---------------------|-----------------|--------------------|")
        
        for cite in sorted(mapped, key=lambda x: x.get("anchor_id", "")):
            anchor_id = cite.get("anchor_id", "")
            mapped_key = cite.get("mapped_ref_key", "N/A")
            strategy = cite.get("strategy_used", "unknown")
            confidence = cite.get("confidence", 0.0)
            
            lines.append(f"| {anchor_id} | {mapped_key} | {strategy} | {confidence:.1f} |")
        
        lines.append("")
    else:
        lines.append("_无已映射引用 (No mapped citations)._")
        lines.append("")
    
    if unmapped:
        lines.append("## 未映射引用 (Unmapped Citations)")
        lines.append("")
        lines.append("| 锚点ID (Anchor ID) | 策略 (Strategy) | 置信度 (Confidence) | 原因 (Reason) |")
        lines.append("|---------------------|-----------------|--------------------|---------------|")
        
        for cite in sorted(unmapped, key=lambda x: x.get("anchor_id", "")):
            anchor_id = cite.get("anchor_id", "")
            strategy = cite.get("strategy_used", "unknown")
            confidence = cite.get("confidence", 0.0)
            reason = cite.get("reason", "unknown")
            
            lines.append(f"| {anchor_id} | {strategy} | {confidence:.1f} | {reason} |")
        
        lines.append("")
    
    # Write the output
    output_path.write_text("\n".join(lines), encoding="utf-8")
    
    # Count sections created
    sections_count = (
        1 +  # Paper Profile
        1 +  # Logic Graph  
        1 +  # Themes
        1 +  # Evidence Index
        1 +  # Key Evidence Lines
        (1 if slots else 0) +  # Figure/Table Slots
        1    # Citations
    )
    
    return sections_count, output_path


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON file, return default if not found."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return default if default is not None else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file, return empty list if not found."""
    if not path.exists():
        return []
    
    results = []
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except IOError:
        pass
    
    return results


def _determine_quality(paper_profile: dict[str, Any], synthesis: dict[str, Any], facts: list[dict[str, Any]], cite_map: list[dict[str, Any]]) -> str:
    """Determine quality label based on data availability."""
    has_profile = bool(paper_profile.get("paper_type") and paper_profile.get("paper_type") != "unknown")
    has_synthesis = bool(synthesis.get("executive_summary") and synthesis.get("key_evidence_lines"))
    has_facts = any(f.get("quote") for f in facts)
    has_citations = any(c.get("mapped_ref_key") for c in cite_map)
    
    quality_scores = sum([has_profile, has_synthesis, has_facts, has_citations])
    
    if quality_scores >= 4:
        return "high"
    elif quality_scores >= 2:
        return "medium"
    else:
        return "low"
