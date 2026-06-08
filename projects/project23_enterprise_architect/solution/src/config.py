"""Configuration for project 23 — Enterprise Compliance Review Agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # LLM
    model: str = field(default_factory=lambda: os.getenv("MODEL", "gpt-4o-mini"))
    litellm_api_base: str | None = field(default_factory=lambda: os.getenv("LITELLM_API_BASE"))

    # Storage
    checkpoint_db: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DB", "data/compliance_checkpoints.db"))
    audit_file: str = field(default_factory=lambda: os.getenv("AUDIT_FILE", "data/audit_trail.jsonl"))

    # Observability
    langfuse_public_key: str | None = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str | None = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    langsmith_api_key: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "compliance-review"))

    # AWS AgentCore
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    agentcore_agent_id: str | None = field(default_factory=lambda: os.getenv("AGENTCORE_AGENT_ID"))

    # Google Vertex AI
    gcp_project_id: str | None = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID"))
    gcp_region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))

    # Business
    analyst_hourly_rate: float = field(default_factory=lambda: float(os.getenv("ANALYST_HOURLY_RATE", "85")))
    manual_hours_per_doc: float = field(default_factory=lambda: float(os.getenv("MANUAL_HOURS_PER_DOC", "5")))

    # Default company policies loaded if MCP server not available
    @property
    def default_policies(self) -> dict[str, str]:
        return {
            "POL-001": "No personally identifiable information (PII) may be stored in unencrypted logs.",
            "POL-002": "All vendor contracts must include a data processing agreement (DPA).",
            "POL-003": "Financial data must be retained for 7 years per SOX requirements.",
            "POL-004": "Contracts exceeding $50,000 require legal review before signing.",
            "POL-005": "GDPR Article 28 compliance is required for all data processors in the EU.",
        }


cfg = Config()
