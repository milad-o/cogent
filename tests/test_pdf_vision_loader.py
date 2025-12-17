from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agenticflow.core.messages import AIMessage
from agenticflow.document.loaders.handlers.pdf_vision import PDFVisionLoader
from agenticflow.models.base import BaseChatModel


@pytest.mark.asyncio
async def test_pdf_vision_loader_sends_multimodal_content(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    # Create a tiny one-page PDF
    pdf_path = tmp_path / "one_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    doc.save(str(pdf_path))
    doc.close()

    @dataclass
    class CaptureModel(BaseChatModel):
        response_text: str = "# Title\n\nBody"
        last_messages: list[dict[str, Any]] | None = field(default=None, init=False)

        def _init_client(self) -> None:  # pragma: no cover
            pass

        def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:  # pragma: no cover
            self.last_messages = messages
            return AIMessage(content=self.response_text)

        async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
            self.last_messages = messages
            return AIMessage(content=self.response_text)

        def bind_tools(self, tools: list[Any], *, parallel_tool_calls: bool = True) -> "CaptureModel":
            return self

    model = CaptureModel()
    loader = PDFVisionLoader(model=model, output_format="markdown")

    docs = await loader.load(pdf_path)
    assert len(docs) == 1
    assert "Title" in docs[0].text

    assert model.last_messages is not None
    assert model.last_messages[0]["role"] == "user"

    content = model.last_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_pdf_vision_loader_json_format_roundtrips(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "one_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello")
    doc.save(str(pdf_path))
    doc.close()

    @dataclass
    class JsonModel(BaseChatModel):
        def _init_client(self) -> None:  # pragma: no cover
            pass

        def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:  # pragma: no cover
            raise RuntimeError("not used")

        async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
            return AIMessage(
                content=json.dumps(
                    {
                        "content": "Hello from model",
                        "header": "H",
                        "footer": "F",
                        "printed_page_number": "1",
                    }
                )
            )

        def bind_tools(self, tools: list[Any], *, parallel_tool_calls: bool = True) -> "JsonModel":
            return self

    loader = PDFVisionLoader(model=JsonModel(), output_format="json")
    docs = await loader.load(
        pdf_path,
        include_headers=True,
        include_footers=True,
        include_printed_page_number=True,
    )

    payload = json.loads(docs[0].text)
    assert payload["content"] == "Hello from model"
    assert payload["header"] == "H"
    assert payload["footer"] == "F"
    assert payload["printed_page_number"] == "1"

    assert docs[0].metadata["header"] == "H"
    assert docs[0].metadata["footer"] == "F"
    assert docs[0].metadata["printed_page_number"] == "1"


@pytest.mark.asyncio
async def test_pdf_vision_loader_pages_option_extracts_specific_page(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "five_pages.pdf"
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), f"PAGE {i+1}")
    doc.save(str(pdf_path))
    doc.close()

    @dataclass
    class CountingModel(BaseChatModel):
        calls: int = field(default=0, init=False)
        last_messages: list[dict[str, Any]] | None = field(default=None, init=False)

        def _init_client(self) -> None:  # pragma: no cover
            pass

        def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:  # pragma: no cover
            raise RuntimeError("not used")

        async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
            self.calls += 1
            self.last_messages = messages
            return AIMessage(content=f"Extracted for call {self.calls}")

        def bind_tools(self, tools: list[Any], *, parallel_tool_calls: bool = True) -> "CountingModel":
            return self

    model = CountingModel()
    loader = PDFVisionLoader(model=model, output_format="markdown")

    docs = await loader.load(pdf_path, pages=[4], extract_toc=False)
    assert len(docs) == 1
    assert model.calls == 1
    assert docs[0].metadata["page"] == 4
    assert docs[0].metadata["total_pages"] == 5


@pytest.mark.asyncio
async def test_pdf_vision_loader_html_output_format(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "one_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello")
    doc.save(str(pdf_path))
    doc.close()

    @dataclass
    class HtmlModel(BaseChatModel):
        def _init_client(self) -> None:  # pragma: no cover
            pass

        def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:  # pragma: no cover
            raise RuntimeError("not used")

        async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
            # Ensure the prompt requests HTML
            prompt_text = messages[0]["content"][0]["text"]
            assert "semantic HTML" in prompt_text
            return AIMessage(content="<h1>Title</h1><p>Body</p>")

        def bind_tools(self, tools: list[Any], *, parallel_tool_calls: bool = True) -> "HtmlModel":
            return self

    loader = PDFVisionLoader(model=HtmlModel(), output_format="html")
    docs = await loader.load(pdf_path, extract_toc=False)
    assert "<h1>Title</h1>" in docs[0].text
    assert docs[0].metadata["output_format"] == "html"


@pytest.mark.asyncio
async def test_pdf_vision_loader_vision_toc_merges_multiple_pages(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    # 3-page PDF; the model will pretend pages 1-2 are TOC, page 3 is not.
    pdf_path = tmp_path / "three_pages.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"PAGE {i+1}")
    doc.save(str(pdf_path))
    doc.close()

    @dataclass
    class TocModel(BaseChatModel):
        toc_calls: int = field(default=0, init=False)

        def _init_client(self) -> None:  # pragma: no cover
            pass

        def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:  # pragma: no cover
            raise RuntimeError("not used")

        async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
            prompt_text = messages[0]["content"][0]["text"]
            is_toc_probe = "Determine if this page is a Table of Contents" in prompt_text

            if is_toc_probe:
                self.toc_calls += 1
                if self.toc_calls == 1:
                    return AIMessage(
                        content=json.dumps(
                            {
                                "is_toc_page": True,
                                "entries": [
                                    {"level": 1, "title": "Intro", "page": 1},
                                ],
                            }
                        )
                    )
                if self.toc_calls == 2:
                    return AIMessage(
                        content=json.dumps(
                            {
                                "is_toc_page": True,
                                "entries": [
                                    {"level": 2, "title": "Overview", "page": 2},
                                ],
                            }
                        )
                    )

                return AIMessage(content=json.dumps({"is_toc_page": False, "entries": []}))

            # Normal page extraction call
            return AIMessage(content="PAGE 3")

        def bind_tools(self, tools: list[Any], *, parallel_tool_calls: bool = True) -> "TocModel":
            return self

    loader = PDFVisionLoader(model=TocModel(), output_format="markdown")
    docs = await loader.load(pdf_path, pages=[3], toc_max_pages=3, extract_toc=True)
    assert len(docs) == 1
    toc = docs[0].metadata.get("toc")
    assert isinstance(toc, list)
    assert {"level": 1, "title": "Intro", "page": 1} in toc
    assert {"level": 2, "title": "Overview", "page": 2} in toc
