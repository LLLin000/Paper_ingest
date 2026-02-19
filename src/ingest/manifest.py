"""Run skeleton and manifest contract implementation.

This module provides the run directory structure and manifest.json creation
for the init-only stage. Per contract in .sisyphus/plans/pdf-blueprint-contracts.md:

Run Layout Contract (lines 14-31):
- run/<id>/pages/
- run/<id>/text/
- run/<id>/vision/
- run/<id>/figures_tables/
- run/<id>/paragraphs/
- run/<id>/citations/
- run/<id>/reading/
- run/<id>/obsidian/
- run/<id>/manifest.json

Manifest Contract (lines 33-42):
Required keys: doc_id, input_pdf_path, input_pdf_sha256, started_at_utc,
               toolchain, model_config, render_config, pipeline_version
"""

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, BaseModel, Field


# Required module directories per Run Layout Contract
RUN_SUBDIRS = [
    "pages",
    "text",
    "vision",
    "figures_tables",
    "paragraphs",
    "citations",
    "reading",
    "obsidian",
]


class ToolchainInfo(BaseModel):
    """Toolchain metadata for reproducibility."""
    python_version: str = Field(
        default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        description="Python version string"
    )
    package_lock_hash: str = Field(
        default="",
        description="Hash of requirements.lock file"
    )


class LLMSettings(BaseModel):
    text_model: str = Field(
        default="",
        description="Text model identifier (placeholder for T7)"
    )
    vision_model: str = Field(
        default="",
        description="Vision model identifier (placeholder for T4)"
    )
    vision_provider: str = Field(
        default="",
        description="Vision model provider (placeholder)"
    )
    reading_provider: str = Field(
        default="",
        description="Reading model provider (placeholder)"
    )
    reading_model: str = Field(
        default="",
        description="Reading model identifier (placeholder)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Model temperature"
    )
    prompt_bundle_version: str = Field(
        default="v1",
        description="Prompt bundle version"
    )


class RenderConfig(BaseModel):
    """Render configuration for page rendering."""
    dpi: int = Field(
        default=150,
        ge=72,
        description="DPI for page rendering"
    )
    scale: float = Field(
        default=2.0,
        ge=0.1,
        description="Scale factor for rendering"
    )


class Manifest(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    input_pdf_path: str = Field(..., description="Path to input PDF file")
    input_pdf_sha256: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA256 hash of input PDF"
    )
    started_at_utc: str = Field(
        ...,
        description="Pipeline start timestamp in ISO 8601 format"
    )
    toolchain: ToolchainInfo = Field(
        default_factory=ToolchainInfo,
        description="Toolchain metadata"
    )
    llm_config: LLMSettings = Field(
        default_factory=LLMSettings,
        serialization_alias="model_config",
        validation_alias=AliasChoices("model_config", "llm_config"),
        description="Model configuration"
    )
    render_config: RenderConfig = Field(
        default_factory=RenderConfig,
        description="Render configuration"
    )
    pipeline_version: str = Field(
        default="0.1.0",
        description="Pipeline version"
    )


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.
    
    Uses chunked reading for memory efficiency with large PDFs.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Lowercase hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_doc_id(pdf_path: Path) -> str:
    """Compute deterministic doc_id from PDF SHA256 prefix.
    
    Per ID Stability Rules (lines 181-184):
    - doc_id deterministic default: SHA256 of input PDF bytes, first 16 hex chars.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        First 16 characters of SHA256 hash
    """
    return compute_sha256(pdf_path)[:16]


def compute_lockfile_hash(lockfile_path: Path) -> str:
    """Compute hash of requirements.lock for toolchain tracking.
    
    Args:
        lockfile_path: Path to requirements.lock
        
    Returns:
        First 16 characters of SHA256 hash, or empty string if not found
    """
    if not lockfile_path.exists():
        return ""
    return compute_sha256(lockfile_path)[:16]


def create_run_skeleton(
    run_root: Path,
    doc_id: str,
) -> Path:
    """Create run directory structure under run/{doc_id}/.
    
    Creates all required module directories per Run Layout Contract.
    
    Args:
        run_root: Root directory for runs (default: "run")
        doc_id: Document identifier
        
    Returns:
        Path to the created run directory
    """
    run_dir = run_root / doc_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir(exist_ok=True)
    
    return run_dir


def create_manifest(
    pdf_path: Path,
    doc_id: Optional[str] = None,
    run_root: Path = Path("run"),
    lockfile_path: Optional[Path] = None,
) -> tuple[Manifest, Path]:
    """Create manifest.json and run skeleton for a document.
    
    This is the primary entry point for the init-only stage.
    
    Args:
        pdf_path: Path to the input PDF file
        doc_id: Optional document identifier (computed from PDF hash if omitted)
        run_root: Root directory for runs
        lockfile_path: Path to requirements.lock (defaults to project root)
        
    Returns:
        Tuple of (Manifest instance, path to run directory)
    """
    # Resolve paths for cross-platform compatibility
    pdf_path = pdf_path.resolve()
    run_root = run_root.resolve()
    
    # Compute deterministic doc_id if not provided
    if doc_id is None:
        doc_id = compute_doc_id(pdf_path)
    
    # Create run skeleton
    run_dir = create_run_skeleton(run_root, doc_id)
    
    # Compute lockfile hash
    if lockfile_path is None:
        # Look for requirements.lock in project root (parent of run_root)
        lockfile_path = run_root.parent / "requirements.lock"
    lock_hash = compute_lockfile_hash(lockfile_path)
    
    # Build manifest
    manifest = Manifest(
        doc_id=doc_id,
        input_pdf_path=str(pdf_path),
        input_pdf_sha256=compute_sha256(pdf_path),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        toolchain=ToolchainInfo(package_lock_hash=lock_hash),
    )
    
    # Write manifest.json
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2, by_alias=True),
        encoding="utf-8"
    )
    
    return manifest, run_dir


def load_manifest(run_dir: Path) -> Manifest:
    """Load manifest.json from a run directory.
    
    Args:
        run_dir: Path to run directory containing manifest.json
        
    Returns:
        Manifest instance
        
    Raises:
        FileNotFoundError: If manifest.json does not exist
        ValueError: If manifest.json is invalid
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    return Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
