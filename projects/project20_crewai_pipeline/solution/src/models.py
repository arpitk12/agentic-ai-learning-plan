"""Pydantic output models for CrewAI tasks."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchReport(BaseModel):
    """Structured output from the Researcher agent."""
    topic: str = Field(description="The researched topic")
    key_facts: list[str] = Field(description="5-10 key facts discovered")
    sources: list[str] = Field(description="URLs or references consulted")
    summary: str = Field(description="300-word executive summary")


class Article(BaseModel):
    """Structured output from the Writer agent."""
    title: str = Field(description="Compelling article title")
    introduction: str = Field(description="Hook paragraph ~150 words")
    body_sections: list[dict] = Field(description="List of {heading, content} dicts")
    conclusion: str = Field(description="Conclusion paragraph ~100 words")
    word_count: int = Field(description="Total word count")


class EditedArticle(BaseModel):
    """Structured output from the Editor agent."""
    title: str
    content: str = Field(description="Full polished article in markdown")
    changes_made: list[str] = Field(description="List of edits applied")
    readability_score: float = Field(description="Flesch-Kincaid score estimate")


class SEOAnalysis(BaseModel):
    """Structured output from the SEO Analyst agent."""
    primary_keyword: str
    secondary_keywords: list[str]
    meta_title: str = Field(max_length=60)
    meta_description: str = Field(max_length=160)
    suggested_headings: list[str]
    seo_score: float = Field(ge=0, le=100)
    recommendations: list[str]
