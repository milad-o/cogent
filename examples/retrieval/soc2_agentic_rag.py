"""Agentic RAG over the sample SOC2 report (tables + summarization).

This example builds a dual-index setup:

1) Detail RAG index (Dense + BM25 via EnsembleRetriever) over Markdown chunks
2) SummaryIndex over full pages, exposed as a *summary* tool (not full-page text)

Why SummaryIndex:
- Standard RAG can only retrieve a handful of chunks, so document-wide summaries
  tend to miss content.
- SummaryIndex generates compact page summaries and lets the agent retrieve many
  relevant summaries for broader coverage.

Usage:
    uv run python examples/retrieval/soc2_agentic_rag.py
    uv run python examples/retrieval/soc2_agentic_rag.py --summaries
    uv run python examples/retrieval/soc2_agentic_rag.py --interactive --summaries
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_embeddings, get_model

from agenticflow.agent import Agent
from agenticflow.document.loaders import PDFMarkdownLoader
from agenticflow.document.splitters import RecursiveCharacterSplitter
from agenticflow.observability.bus import EventBus
from agenticflow.observability.event import EventType
from agenticflow.retriever import BM25Retriever, DenseRetriever, EnsembleRetriever, SummaryIndex
from agenticflow.tools.base import BaseTool
from agenticflow.vectorstore import Document, VectorStore


def _default_pdf_path() -> Path:
    return Path("examples/data/sample_soc2.pdf")


async def _load_pdf_markdown_pages(*, pdf_path: Path) -> list[Document]:
    loader = PDFMarkdownLoader(
        ignore_images=True,
        show_progress=True,
    )
    pages = await loader.load(pdf_path)
    return pages


def _chunk_pages(*, pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        keep_separator=True,
    )
    return splitter.split_documents(pages)


async def _build_detail_retriever(*, chunks: list[Document]) -> EnsembleRetriever:
    vectorstore = VectorStore(embeddings=get_embeddings())
    await vectorstore.add_documents(chunks)

    dense = DenseRetriever(vectorstore=vectorstore)
    bm25 = BM25Retriever()
    bm25.add_documents(chunks)

    return EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[0.6, 0.4],
        fusion="rrf",
    )


async def _build_summary_index(*, pages: list[Document]) -> SummaryIndex:
    model = get_model()
    summary_vectorstore = VectorStore(embeddings=get_embeddings())
    summary_index = SummaryIndex(
        llm=model,
        vectorstore=summary_vectorstore,
        extract_keywords=True,
        extract_entities=False,
    )

    event_bus = EventBus()

    def _print_summary_index_event(event: Any) -> None:
        if event.type == EventType.SUMMARY_INDEX_START:
            print(
                "[SummaryIndex] start",
                f"documents={event.data.get('documents_count')}",
            )
            return

        if event.type == EventType.SUMMARY_INDEX_DOCUMENT_SUMMARIZED:
            doc_index = event.data.get("doc_index")
            doc_total = event.data.get("doc_total")
            page = event.data.get("page")
            duration_ms = event.data.get("duration_ms")

            # Keep output readable: print first, every 5, and last
            should_print = (
                doc_index in (1, doc_total)
                or (isinstance(doc_index, int) and doc_index % 5 == 0)
            )
            if should_print:
                suffix = f" page={page}" if page is not None else ""
                dur = f" {duration_ms:.0f}ms" if isinstance(duration_ms, (int, float)) else ""
                print(f"[SummaryIndex] {doc_index}/{doc_total} summarized{suffix}{dur}")
            return

        if event.type == EventType.SUMMARY_INDEX_COMPLETE:
            print(
                "[SummaryIndex] complete",
                f"duration_ms={event.data.get('duration_ms'):.0f}"
                if isinstance(event.data.get("duration_ms"), (int, float))
                else "",
            )
            return

        if event.type == EventType.SUMMARY_INDEX_ERROR:
            print(
                "[SummaryIndex] error",
                f"error={event.data.get('error')}",
            )

    event_bus.subscribe_many(
        [
            EventType.SUMMARY_INDEX_START,
            EventType.SUMMARY_INDEX_DOCUMENT_SUMMARIZED,
            EventType.SUMMARY_INDEX_COMPLETE,
            EventType.SUMMARY_INDEX_ERROR,
        ],
        _print_summary_index_event,
    )
    summary_index.event_bus = event_bus

    await summary_index.add_documents(pages)
    return summary_index


def _make_summary_tool(*, summary_index: SummaryIndex) -> BaseTool:
    async def _search_soc2_page_summaries(
        query: str,
        k: int = 20,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = await summary_index.retrieve_with_scores(query, k=k, filter=filter)
        payload: list[dict[str, Any]] = []
        for r in results:
            # SummaryIndex stores the LLM summary in RetrievalResult.metadata
            summary = r.metadata.get("summary")
            keywords = r.metadata.get("keywords", [])
            page = r.document.metadata.get("page")
            payload.append(
                {
                    "page": page,
                    "summary": summary,
                    "keywords": keywords,
                    "score": r.score,
                }
            )
        return payload

    return BaseTool(
        name="search_soc2_page_summaries",
        description=(
            "Search LLM-generated *page summaries* of the SOC2 report. "
            "Use this for overviews, structure questions, and summarization across many pages. "
            "Returns page numbers + concise summaries + keywords."
        ),
        func=_search_soc2_page_summaries,
        args_schema={
            "query": {"type": "string", "description": "Search query."},
            "k": {
                "type": "integer",
                "description": "Number of page summaries to return.",
                "default": 20,
                "minimum": 1,
            },
            "filter": {
                "type": "object",
                "description": "Optional metadata filter.",
                "additionalProperties": True,
            },
        },
    )


def _create_soc2_agent(*, detail_retriever: EnsembleRetriever, summary_index: SummaryIndex | None) -> Agent:
    detail_tool = detail_retriever.as_tool(
        name="search_soc2_details",
        description=(
            "Search the SOC2 report for specific details (controls, tests, tables, exact wording). "
            "Returns relevant passages with page metadata. Best for precise questions and table lookups."
        ),
        k_default=6,
        include_scores=True,
        include_metadata=True,
    )

    tools: list[Any] = [detail_tool]
    if summary_index is not None:
        tools.append(_make_summary_tool(summary_index=summary_index))

    instructions = """You are a SOC 2 compliance analyst.

You have tools:
- search_soc2_details: best for precise facts, controls, and tables.
- search_soc2_page_summaries: best for overviews and summarization (if available).

Rules:
1) For table questions, use search_soc2_details and present relevant table rows/columns in Markdown.
2) For summarization/overview questions, use search_soc2_page_summaries (retrieve many summaries) and synthesize.
3) Cite page numbers whenever possible.
4) If the report doesn't contain the answer, say so explicitly.
"""

    return Agent(
        name="SOC2Analyst",
        model=get_model(),
        tools=tools,
        instructions=instructions,
        memory=True,
    )


async def _run_non_interactive(*, agent: Agent, thread_id: str) -> None:
    questions = [
        "What time period does this SOC2 report cover? Cite the page.",
        "Find any table that summarizes controls or tests performed and show it in markdown.",
    ]
    questions_with_summaries = [
        "Give an executive summary of this SOC2 report.",
        "What are the major sections or themes in the report?",
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        print("=" * 80)
        print(await agent.run(q, thread_id=thread_id))

    # If SummaryIndex tool exists, these are more useful
    if any(t.name == "search_soc2_page_summaries" for t in agent.config.tools):
        for q in questions_with_summaries:
            print("\n" + "=" * 80)
            print(f"Q: {q}")
            print("=" * 80)
            print(await agent.run(q, thread_id=thread_id))


async def _run_interactive(*, agent: Agent, thread_id: str) -> None:
    print("\nInteractive mode. Enter a question (blank to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        print("\n" + (await agent.run(q, thread_id=thread_id)))


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG over the sample SOC2 report")
    parser.add_argument("--pdf", type=Path, default=_default_pdf_path())
    parser.add_argument("--summaries", action="store_true", help="Enable SummaryIndex + summaries tool")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A loop")
    parser.add_argument("--thread-id", type=str, default="soc2-demo")
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("=" * 80)
    print(f"Loading: {pdf_path}")
    print("=" * 80)

    pages = await _load_pdf_markdown_pages(pdf_path=pdf_path)
    print(f"Loaded {len(pages)} pages")

    chunks = _chunk_pages(pages=pages)
    print(f"Created {len(chunks)} chunks")

    detail_retriever = await _build_detail_retriever(chunks=chunks)

    summary_index: SummaryIndex | None = None
    if args.summaries:
        print("Building SummaryIndex (LLM calls per page)...")
        summary_index = await _build_summary_index(pages=pages)
        print(f"SummaryIndex ready: {len(summary_index.summaries)} summaries")

    agent = _create_soc2_agent(detail_retriever=detail_retriever, summary_index=summary_index)

    if args.interactive:
        await _run_interactive(agent=agent, thread_id=args.thread_id)
    else:
        await _run_non_interactive(agent=agent, thread_id=args.thread_id)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
