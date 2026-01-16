"""Hierarchical index for structured document retrieval.

Organizes documents into multiple levels of granularity:
- Document level (full document summary)
- Section level (chapters, sections)
- Paragraph level (logical paragraphs)
- Chunk level (small retrieval units)

Retrieval works top-down: find relevant sections first, then drill into chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.retriever.utils.llm_adapter import adapt_llm
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.models import Model
    from agenticflow.retriever.utils.llm_adapter import LLMProtocol
    from agenticflow.vectorstore import VectorStore


class HierarchyLevel(Enum):
    """Hierarchy levels in document structure."""
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    CHUNK = "chunk"


@dataclass
class HierarchyNode:
    """A node in the document hierarchy."""

    node_id: str
    level: HierarchyLevel
    title: str
    text: str
    summary: str | None = None
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Position in document
    start_pos: int = 0
    end_pos: int = 0
    depth: int = 0  # 0 = document, 1 = section, etc.


class HierarchicalIndex(BaseRetriever):
    """Index that respects and leverages document structure.

    Parses document structure (Markdown headers, HTML tags, or custom markers)
    and builds a hierarchy. Retrieval first finds relevant high-level sections,
    then drills down to specific chunks.

    Benefits:
    - Respects document organization
    - Reduces noise from irrelevant sections
    - Provides context at multiple levels
    - Efficient for long, structured documents

    Example:
        ```python
        from agenticflow.retriever import HierarchicalIndex

        index = HierarchicalIndex(
            vectorstore=vs,
            llm=model,  # Optional: for section summaries
            structure_type="markdown",  # or "html", "custom"
        )

        await index.add_documents(docs)

        # Retrieval finds section first, then relevant chunks
        results = await index.retrieve("installation instructions")

        # Results include hierarchy context
        for r in results:
            print(f"Section: {r.metadata['section_title']}")
            print(f"Path: {r.metadata['hierarchy_path']}")
            print(f"Content: {r.document.text}")
        ```
    """

    _name: str = "hierarchical"

    # Markdown header pattern
    MD_HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    # HTML header pattern
    HTML_HEADER_PATTERN = re.compile(r'<h([1-6])[^>]*>(.+?)</h\1>', re.IGNORECASE | re.DOTALL)

    SUMMARIZE_SECTION_PROMPT = '''Summarize this section in 1-2 sentences, capturing the main topic.

Section Title: {title}
Content:
{text}

Summary:'''

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: Model | None = None,
        *,
        structure_type: str = "markdown",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        generate_summaries: bool = True,
        top_k_sections: int = 3,
        chunks_per_section: int = 3,
        name: str | None = None,
    ) -> None:
        """Create a hierarchical index.

        Args:
            vectorstore: Vector store for embeddings.
            llm: Language model for section summaries (optional).
            structure_type: "markdown", "html", or "custom".
            chunk_size: Size of leaf chunks.
            chunk_overlap: Overlap between chunks.
            generate_summaries: Generate LLM summaries for sections.
            top_k_sections: Number of sections to consider.
            chunks_per_section: Chunks to retrieve per section.
            name: Optional custom name.
        """
        self._vectorstore = vectorstore
        self._llm: LLMProtocol | None = adapt_llm(llm) if llm is not None else None
        self._structure_type = structure_type
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._generate_summaries = generate_summaries and llm is not None
        self._top_k_sections = top_k_sections
        self._chunks_per_section = chunks_per_section

        # Storage
        self._nodes: dict[str, HierarchyNode] = {}
        self._documents: dict[str, Document] = {}
        self._section_nodes: list[str] = []  # For section-level search
        self._chunk_nodes: list[str] = []     # For chunk-level search

        if name:
            self._name = name

    def _generate_id(self, text: str, prefix: str = "") -> str:
        """Generate unique node ID."""
        hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_val}" if prefix else hash_val

    def _parse_markdown_structure(self, text: str, doc_id: str) -> list[HierarchyNode]:
        """Parse Markdown headers into hierarchy."""
        nodes: list[HierarchyNode] = []

        # Find all headers with positions
        headers: list[tuple[int, int, str, int]] = []  # (level, pos, title, end_pos)

        for match in self.MD_HEADER_PATTERN.finditer(text):
            level = len(match.group(1))  # Number of #
            title = match.group(2).strip()
            pos = match.start()
            headers.append((level, pos, title, 0))

        # Calculate end positions
        for i, (level, pos, title, _) in enumerate(headers):
            end_pos = headers[i + 1][1] if i + 1 < len(headers) else len(text)
            headers[i] = (level, pos, title, end_pos)

        # Build hierarchy
        parent_stack: list[str] = []  # Stack of parent IDs by level

        for level, pos, title, end_pos in headers:
            section_text = text[pos:end_pos].strip()

            # Determine hierarchy level
            if level == 1:
                hier_level = HierarchyLevel.SECTION
            elif level == 2:
                hier_level = HierarchyLevel.SUBSECTION
            else:
                hier_level = HierarchyLevel.PARAGRAPH

            node_id = self._generate_id(f"{doc_id}_{title}_{pos}", "sec")

            # Find parent (closest header with lower level)
            parent_id = None
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()
            if parent_stack:
                parent_id = parent_stack[-1][1]

            node = HierarchyNode(
                node_id=node_id,
                level=hier_level,
                title=title,
                text=section_text,
                parent_id=parent_id,
                start_pos=pos,
                end_pos=end_pos,
                depth=level,
                metadata={"doc_id": doc_id, "header_level": level},
            )
            nodes.append(node)

            # Update parent's children
            if parent_id:
                for n in nodes:
                    if n.node_id == parent_id:
                        n.children.append(node_id)
                        break

            parent_stack.append((level, node_id))

        return nodes

    def _parse_html_structure(self, text: str, doc_id: str) -> list[HierarchyNode]:
        """Parse HTML headers into hierarchy."""
        nodes: list[HierarchyNode] = []

        for match in self.HTML_HEADER_PATTERN.finditer(text):
            level = int(match.group(1))
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            pos = match.start()

            hier_level = HierarchyLevel.SECTION if level <= 2 else HierarchyLevel.SUBSECTION

            node_id = self._generate_id(f"{doc_id}_{title}_{pos}", "sec")
            node = HierarchyNode(
                node_id=node_id,
                level=hier_level,
                title=title,
                text=title,  # HTML parsing would need more context
                start_pos=pos,
                depth=level,
                metadata={"doc_id": doc_id, "header_level": level},
            )
            nodes.append(node)

        return nodes

    def _chunk_text(self, text: str, node_id: str, doc_id: str) -> list[HierarchyNode]:
        """Split text into chunk nodes."""
        chunks: list[HierarchyNode] = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "\n"]:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self._chunk_size // 2:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        break

            if chunk_text.strip():
                chunk_id = self._generate_id(f"{node_id}_chunk_{chunk_idx}", "chunk")
                chunks.append(HierarchyNode(
                    node_id=chunk_id,
                    level=HierarchyLevel.CHUNK,
                    title=f"Chunk {chunk_idx}",
                    text=chunk_text.strip(),
                    parent_id=node_id,
                    start_pos=start,
                    end_pos=start + len(chunk_text),
                    depth=99,  # Leaf level
                    metadata={"doc_id": doc_id, "chunk_idx": chunk_idx},
                ))
                chunk_idx += 1

            start += len(chunk_text) - self._chunk_overlap
            if start <= 0:
                start = end

        return chunks

    async def _generate_summary(self, node: HierarchyNode) -> str:
        """Generate summary for a section."""
        if not self._llm:
            return node.text[:200]

        prompt = self.SUMMARIZE_SECTION_PROMPT.format(
            title=node.title,
            text=node.text[:3000],
        )
        return await self._llm.generate(prompt)

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and build hierarchy.

        Args:
            documents: Documents to index.

        Returns:
            List of document IDs.
        """
        doc_ids = []
        section_texts: list[str] = []
        section_ids: list[str] = []
        chunk_texts: list[str] = []
        chunk_ids: list[str] = []

        for doc in documents:
            doc_id = doc.id or self._generate_id(doc.text, "doc")
            self._documents[doc_id] = doc
            doc_ids.append(doc_id)

            # Create document-level node
            doc_node = HierarchyNode(
                node_id=doc_id,
                level=HierarchyLevel.DOCUMENT,
                title=doc.metadata.get("title", "Document"),
                text=doc.text,
                depth=0,
                metadata=doc.metadata,
            )
            self._nodes[doc_id] = doc_node

            # Parse structure
            if self._structure_type == "markdown":
                section_nodes = self._parse_markdown_structure(doc.text, doc_id)
            elif self._structure_type == "html":
                section_nodes = self._parse_html_structure(doc.text, doc_id)
            else:
                # No structure - treat as single section
                section_nodes = [HierarchyNode(
                    node_id=self._generate_id(doc_id, "sec"),
                    level=HierarchyLevel.SECTION,
                    title="Content",
                    text=doc.text,
                    parent_id=doc_id,
                    depth=1,
                    metadata={"doc_id": doc_id},
                )]

            # Add section nodes and generate summaries
            for section in section_nodes:
                if section.parent_id is None:
                    section.parent_id = doc_id
                    doc_node.children.append(section.node_id)

                # Generate summary if enabled
                if self._generate_summaries:
                    section.summary = await self._generate_summary(section)

                self._nodes[section.node_id] = section
                self._section_nodes.append(section.node_id)

                # Use summary or title+text for embedding
                embed_text = section.summary or f"{section.title}\n{section.text[:500]}"
                section_texts.append(embed_text)
                section_ids.append(section.node_id)

                # Create chunks for this section
                chunks = self._chunk_text(section.text, section.node_id, doc_id)
                for chunk in chunks:
                    self._nodes[chunk.node_id] = chunk
                    self._chunk_nodes.append(chunk.node_id)
                    section.children.append(chunk.node_id)

                    chunk_texts.append(chunk.text)
                    chunk_ids.append(chunk.node_id)

        # Add to vector store with level metadata
        if section_texts:
            await self._vectorstore.add_texts(
                section_texts,
                metadatas=[{"node_id": nid, "level": "section"} for nid in section_ids],
            )

        if chunk_texts:
            await self._vectorstore.add_texts(
                chunk_texts,
                metadatas=[{"node_id": nid, "level": "chunk"} for nid in chunk_ids],
            )

        return doc_ids

    def _get_hierarchy_path(self, node_id: str) -> list[str]:
        """Get path from root to node."""
        path = []
        current = node_id

        while current and current in self._nodes:
            node = self._nodes[current]
            path.append(node.title)
            current = node.parent_id

        return path[::-1]  # Root to leaf

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using hierarchical search.

        First finds relevant sections, then retrieves chunks from those sections.
        """
        results: list[RetrievalResult] = []

        # Step 1: Find relevant sections
        section_filter = {"level": "section"}
        if filter:
            section_filter.update(filter)

        section_results = await self._vectorstore.search(
            query, k=self._top_k_sections, filter=section_filter
        )

        relevant_section_ids = set()
        for sr in section_results:
            node_id = sr.document.metadata.get("node_id")
            if node_id:
                relevant_section_ids.add(node_id)

        # Step 2: Get chunks from relevant sections
        if relevant_section_ids:
            # Search within section chunks
            chunk_results = await self._vectorstore.search(
                query, k=k * 2, filter={"level": "chunk"}
            )

            for cr in chunk_results:
                node_id = cr.document.metadata.get("node_id")
                if node_id and node_id in self._nodes:
                    chunk_node = self._nodes[node_id]

                    # Check if chunk is in a relevant section
                    if chunk_node.parent_id in relevant_section_ids:
                        section_node = self._nodes.get(chunk_node.parent_id)

                        results.append(RetrievalResult(
                            document=Document(
                                text=chunk_node.text,
                                metadata={
                                    **chunk_node.metadata,
                                    "node_id": node_id,
                                },
                            ),
                            score=cr.score,
                            retriever_name=self.name,
                            metadata={
                                "section_title": section_node.title if section_node else "",
                                "section_summary": section_node.summary if section_node else "",
                                "hierarchy_path": self._get_hierarchy_path(node_id),
                                "level": "chunk",
                            },
                        ))

                        if len(results) >= k:
                            break

        # Fallback: direct chunk search if no sections matched
        if not results:
            chunk_results = await self._vectorstore.search(query, k=k)
            for cr in chunk_results:
                node_id = cr.document.metadata.get("node_id")
                results.append(RetrievalResult(
                    document=cr.document,
                    score=cr.score,
                    retriever_name=self.name,
                    metadata={
                        "hierarchy_path": self._get_hierarchy_path(node_id) if node_id else [],
                        "level": cr.document.metadata.get("level", "unknown"),
                    },
                ))

        return results[:k]

    def get_structure(self, doc_id: str | None = None) -> dict[str, Any]:
        """Get the document structure as a tree.

        Args:
            doc_id: Optional document ID. If None, returns all docs.

        Returns:
            Nested dictionary representing document hierarchy.
        """
        def build_tree(node_id: str) -> dict[str, Any]:
            node = self._nodes.get(node_id)
            if not node:
                return {}

            return {
                "id": node.node_id,
                "title": node.title,
                "level": node.level.value,
                "summary": node.summary,
                "children": [build_tree(cid) for cid in node.children],
            }

        if doc_id:
            return build_tree(doc_id)

        return {
            "documents": [
                build_tree(nid)
                for nid, node in self._nodes.items()
                if node.level == HierarchyLevel.DOCUMENT
            ]
        }
