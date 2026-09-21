"""Summarizer behavior: deterministic truncation and LLM caching."""

from __future__ import annotations

from jet.context.summarizer import LLMSummarizer, TruncatingSummarizer
from tests.conftest import make_llm


async def test_truncating_summarizer_respects_target() -> None:
    summarizer = TruncatingSummarizer()
    text = "word " * 500
    short = await summarizer.summarize(text, target_tokens=10, style="short")
    long = await summarizer.summarize(text, target_tokens=50, style="long")
    assert len(short) < len(long) < len(text)


async def test_llm_summarizer_uses_provider_and_caches() -> None:
    llm = make_llm("condensed memory", "second summary")
    summarizer = LLMSummarizer(llm)
    first = await summarizer.summarize("original text", target_tokens=20, style="long")
    second = await summarizer.summarize("original text", target_tokens=20, style="long")
    assert first == second == "condensed memory"
    assert len(llm.calls) == 1


async def test_llm_summarizer_distinguishes_styles() -> None:
    llm = make_llm("long summary", "short note")
    summarizer = LLMSummarizer(llm)
    long = await summarizer.summarize("text", target_tokens=20, style="long")
    short = await summarizer.summarize("text", target_tokens=20, style="short")
    assert (long, short) == ("long summary", "short note")
    assert len(llm.calls) == 2


async def test_empty_text_returns_empty_without_calling_llm() -> None:
    llm = make_llm("should not be used")
    summarizer = LLMSummarizer(llm)
    assert await summarizer.summarize("   ", target_tokens=10) == ""
    assert llm.calls == []
