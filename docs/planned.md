Let me check what capabilities we currently have and what might be missing: 

Here's what we currently have:

| Capability | Purpose |
|------------|---------|
| **KnowledgeGraph** | Entity/relationship memory, multi-hop reasoning |
| **CodebaseAnalyzer** | Python AST parsing, code exploration |
| **FileSystem** | Sandboxed file operations |
| **WebSearch** | DuckDuckGo search, page fetching |
| **CodeSandbox** | Safe Python code execution |
| **SSISAnalyzer** | SSIS package analysis (domain-specific) |

**Common capabilities we're missing:**

1. **Database** - SQL query execution, schema exploration, data analysis
2. **API/HTTP** - Make HTTP requests, call REST APIs, handle auth
3. **Email** - Send/read emails (SMTP/IMAP)
4. **Calendar** - Schedule management, event creation
5. **Shell/Terminal** - Execute shell commands safely
6. **Image/Vision** - Image analysis, generation, OCR
7. **Audio/Speech** - Text-to-speech, speech-to-text
8. **PDF** - Read/generate PDF documents
9. **Spreadsheet** - Excel/CSV manipulation
10. **Vector Store** - RAG with embeddings (Chroma, Pinecone, etc.)
11. **Browser/Playwright** - Web automation, scraping with JS rendering
12. **Slack/Discord** - Chat platform integration
13. **Git** - Repository operations, commits, branches

**Most impactful additions** for a general-purpose agent framework:

| Priority | Capability | Why |
|----------|------------|-----|
| 🔴 High | **Database** | Essential for data apps |
| 🔴 High | **HTTP/API** | Connect to any service |
| 🔴 High | **VectorStore** | RAG is core to modern AI apps |
| 🟡 Medium | **Shell** | System automation |
| 🟡 Medium | **Browser** | Web automation beyond search |
| 🟢 Nice | **PDF/Docs** | Document processing |