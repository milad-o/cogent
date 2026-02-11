# Graph Module (/graph/) - Build Taskboard

**Status:** 🔨 IN PROGRESS (Phase 1 ✅ Phase 2 ✅ Phase 3A ✅ Phase 3B ✅ Phase 4 ✅ + Refactoring ✅ Phase 6 ✅)  
**Approach:** Build from scratch - **NO CODE COPYING**  
**Timeline:** 19-27 hours (~3-4 days)

**Architecture:** 2-layer design - **Engine (how) + Storage (where)**  
**Design:** **Async-first API**, direct methods (no chaining), modern Python 3.13+

**Core Philosophy:**
- **Graph = Engine + Storage** - Fully composable, mix and match
- **Smart defaults** - Works out of the box with `Graph()`
- **SQL is storage** - Not a bundled backend, works with any engine
- **Bundled backends** - Only for native graph DBs (Neo4j, FalkorDB)

---

## 🏗️ Architecture Clarity

### Two-Layer Design

```
┌─────────────────────────────────────┐
│      ENGINE LAYER                   │  ← How: Graph operations & algorithms
│  (NativeEngine, NetworkXEngine)     │
├─────────────────────────────────────┤
│      STORAGE LAYER                  │  ← Where: Data persistence
│  (MemoryStorage, FileStorage, SQL)  │
└─────────────────────────────────────┘
```

**Key Insight:** Engine and Storage are COMPLETELY INDEPENDENT!

### Component Taxonomy

**Separation of Concerns (after refactoring):**
- **Engine:** Graph topology & algorithms (find_path, neighbors, counts)
- **Storage:** Data persistence & CRUD (entities, relationships, stats)

| Component | Engine | Storage | Separable? | Example |
|-----------|--------|---------|------------|---------|
| **Graph (base)** | Any | Any | ✅ Yes | `Graph(engine=NetworkX, storage=SQL)` |
| **Neo4jGraph** | Neo4j only | Neo4j only | ❌ No | `Neo4jGraph(uri=..., auth=...)` |
| **FalkorDBGraph** | FalkorDB only | Redis only | ❌ No | `FalkorDBGraph(host=...)` |

**Engine Options:**
- **NativeEngine** - Pure Python (dict/list), zero dependencies
- **NetworkXEngine** - 100+ graph algorithms (default if available)
- **Future:** RustworkxEngine, CypherEngine

**Storage Options:**
- **MemoryStorage** - Transient, fast (default)
- **FileStorage** - JSON/pickle persistence
- **SQLStorage** - Any SQL database (PostgreSQL, MySQL, SQLite) via SQLAlchemy v2 ORM with DRY CRUD
- **Future:** Neo4jStorage (for Neo4jGraph only)

### Design Principles

1. **Engine ≠ Storage** - Complete separation of concerns
2. **Graph is base class** - Not KnowledgeGraph
3. **No chaining** - Direct methods, Pythonic API
4. **No string queries** - Dict/dataclass patterns for complex queries
5. **Smart defaults** - `Graph()` works out of the box

---

## 📋 Phase 1: Core Data Models (1-2h)
**Status:** ✅ COMPLETE  
**Goal:** Clean domain models from scratch

- [x] Create `src/cogent/graph/` directory
- [x] Create `graph/models.py`
- [x] Write `Entity` dataclass (id, type, attributes, timestamps)
- [x] Write `Relationship` dataclass (source, relation, target, attributes)
- [x] Add validation logic (non-empty IDs, valid types)
- [x] Add comprehensive docstrings
- [x] Full type hints (Python 3.13+)
- [x] Create `tests/graph/` directory
- [x] Write `tests/graph/test_models.py`
- [x] ✅ Run tests → all passing (30/30 tests)

**Deliverable:** `graph/models.py` (209 lines) + tests (317 lines) = **535 lines total**

---

## 📋 Phase 2: Storage Protocol (1h)
**Status:** ✅ COMPLETE  
**Goal:** Define async backend contract with bulk operations

- [x] Create `graph/storage.py`
- [x] Write `StorageBackend` protocol class
- [x] Convert all methods to async (async-first architecture)
- [x] Define method: `async add_entity(id, type, attrs) -> Entity`
- [x] Define method: `async add_entities(entities) -> list[Entity]` **BULK**
- [x] Define method: `async get_entity(id) -> Entity | None`
- [x] Define method: `async remove_entity(id) -> bool`
- [x] Define method: `async add_relationship(source, relation, target, attrs) -> Relationship`
- [x] Define method: `async add_relationships(relationships) -> list[Relationship]` **BULK**
- [x] Define method: `async get_relationships(entity_id) -> list[Relationship]`
- [x] Define method: `async query(pattern) -> list[dict]`
- [x] Define method: `async find_path(source, target) -> list[str] | None`
- [x] Define method: `async get_all_entities() -> list[Entity]`
- [x] Define method: `async stats() -> dict[str, int]`
- [x] Define method: `async clear() -> None`
- [x] Add full docstrings for each method with async examples
- [x] Write `tests/graph/test_storage.py` with async tests
- [x] ✅ Run tests → all 22 tests passing (20 + 2 bulk tests)

**Deliverable:** `graph/storage.py` (263 lines) + tests (407 lines) = **670 lines total**

---

## 📋 Phase 3A: Engine Implementations (3-4h)
**Status:** ✅ COMPLETE  
**Goal:** Build engine layer - how graph operations work

**Engines:**
- **NativeEngine** - Pure Python (dict/list), zero dependencies
- **NetworkXEngine** - Graph library with 100+ algorithms

- [x] Create `graph/engines/` directory
- [x] Create `graph/engines/__init__.py`
- [x] Create `graph/engines/base.py`
- [x] Write `Engine` protocol class
- [x] Define methods: `add_node`, `remove_node`, `get_node`
- [x] Define methods: `add_edge`, `remove_edge`, `get_edges`
- [x] Define methods: `get_neighbors`, `find_path`, `get_subgraph`
- [x] Define methods: `shortest_path`, `connected_components`
- [x] Add comprehensive docstrings

**NativeEngine Implementation:**
- [x] Create `graph/engines/native.py`
- [x] Write `NativeEngine` class (implements Engine protocol)
- [x] Use `_nodes: dict[str, dict]` for node storage
- [x] Use `_edges: dict[str, list[str]]` for adjacency lists
- [x] Implement `add_node()`, `remove_node()`, `get_node()`
- [x] Implement `add_edge()`, `remove_edge()`, `get_edges()`
- [x] Implement `get_neighbors()` - adjacency list lookup
- [x] Implement `find_path()` - BFS from scratch
- [x] Implement `get_subgraph()` - node filtering
- [x] Pure Python, zero dependencies

**NetworkXEngine Implementation:**
- [x] Create `graph/engines/networkx.py`
- [x] Write `NetworkXEngine` (implements Engine protocol)
- [x] Use `nx.DiGraph()` with optional dependency check
- [x] Implement all Engine methods (wrap NetworkX API)
- [x] Leverage 100+ NetworkX algorithms
- [x] Fallback to NativeEngine if NetworkX unavailable

- [x] Write `tests/graph/test_engines.py`
- [x] Test NativeEngine (always available)
- [x] Test NetworkXEngine (if NetworkX installed)
- [x] Test protocol conformance
- [x] Test path finding algorithms
- [x] Test engine selection logic
- [x] ✅ Run tests → all 72 tests passing

**Deliverables:**
- `graph/engines/__init__.py` (16 lines)
- `graph/engines/base.py` (256 lines)
- `graph/engines/native.py` (259 lines)
- `graph/engines/networkx.py` (283 lines)
- `tests/graph/test_engines.py` (519 lines)
- **Total: 1,333 lines**

---

## 📋 Phase 3B: Storage Implementations (3-4h)
**Status:** ✅ COMPLETE  
**Goal:** Build storage layer - where data lives

**Storage:**
- **MemoryStorage** - In-memory, transient (default)
- **FileStorage** - JSON/pickle persistence
- **SQLStorage** - Any SQL database (PostgreSQL, MySQL, SQLite)

- [x] Refactor `graph/storage.py` → `graph/storage/base.py`
  - [x] Keep async Storage protocol (renamed from StorageBackend)
  - [x] Move file to storage/ directory
  - [x] Update imports in `__init__.py`
- [x] Create `graph/storage/` directory
- [x] Create `graph/storage/__init__.py`

**MemoryStorage Implementation:**
- [x] Create `graph/storage/memory.py`
- [x] Write `MemoryStorage` (implements Storage protocol)
- [x] Use dicts/lists: `_entities: dict[str, Entity]`, `_relationships: list[Relationship]`
- [x] Implement all Storage methods (in-memory operations)
- [x] Fast, transient (data lost on exit)

**FileStorage Implementation:**
- [x] Create `graph/storage/file.py`
- [x] Write `FileStorage` (implements Storage protocol)
- [x] Constructor: `__init__(path: str, format: "json" | "pickle")`
- [x] Support JSON (human-readable) and pickle (faster)
- [x] Auto-load on init, auto-save on modifications
- [x] Implement all Storage methods with error handling

**SQLStorage Implementation:**
- [x] Install dependencies: `uv add sqlalchemy aiosqlite asyncpg`
- [x] Create `graph/storage/sql.py` (SQLAlchemy v2 ORM + DRY CRUD pattern)
- [x] Constructor: `__init__(connection_string: str)` (DB-agnostic)
- [x] Define models: `EntityModel`, `RelationshipModel` (JSON attributes)
- [x] Setup: `create_async_engine()`, `async_sessionmaker()`, create tables with indexes
- [x] Implement generic CRUD base: `_get_by_id[T]`, `_create[T]`, `_create_many[T]`, `_delete[T]`
- [x] Implement all Storage methods (reuse generic CRUD)
- [x] Special: `find_path()` with BFS (recursive CTE for PostgreSQL as TODO)
- [x] Transaction support, any SQL DB, in-memory option (`sqlite:///:memory:`)

- [x] Write `tests/graph/test_storage_impls.py`
- [x] Test MemoryStorage (all methods)
- [x] Test FileStorage (JSON and pickle, persistence)
- [x] Test SQLStorage (SQLite in-memory for CI)
- [x] Test protocol conformance for all three
- [x] Test persistence (file and SQL)
- [x] Test bulk operations, cascade delete, error handling
- [x] ✅ Run tests → all 162 tests passing (39 new storage tests)

**Deliverables:**
- `graph/storage/base.py` (501 lines - refactored from storage.py)
- `graph/storage/memory.py` (211 lines)
- `graph/storage/file.py` (320 lines)
- `graph/storage/sql.py` (443 lines)
- `graph/storage/__init__.py` (12 lines)
- `tests/graph/test_storage_impls.py` (630 lines)
- **Total: 2,117 lines**

---

## 📋 Phase 4: Base Graph Class (2-3h)
**Status:** ✅ COMPLETE  
**Goal:** Composable Graph with engine + storage

- [x] Create `graph/graph.py`
- [x] Write `Graph` class (async context manager)
- [x] Constructor: `__init__(engine: Engine | None = None, storage: Storage | None = None)`
  - [x] Smart defaults: NetworkX (if available, else Native) + MemoryStorage
  - [x] Fallback logic for missing NetworkX
- [x] Delegate operations to engine AND storage:
  - [x] Engine handles graph algorithms and traversal
  - [x] Storage handles persistence and retrieval
  - [x] Coordinate between them (sync state)
- [x] Implement core methods:
  - [x] `async add_entity(id, type, **attrs) -> Entity`
  - [x] `async get_entity(id) -> Entity | None`
  - [x] `async remove_entity(id) -> bool`
  - [x] `async add_relationship(source, relation, target, **attrs) -> Relationship`
  - [x] `async get_relationships(entity_id) -> list[Relationship]`
  - [x] `async add_entities(entities) -> list[Entity]` (bulk)
  - [x] `async add_relationships(rels) -> list[Relationship]` (bulk)
- [x] Implement query methods:
  - [x] `async find_entities(type=..., attributes=...) -> list[Entity]`
  - [x] `async find_path(start, end) -> list[str] | None`
  - [x] `async get_neighbors(entity_id) -> list[str]`
- [x] Implement utility methods:
  - [x] `async get_all_entities() -> list[Entity]`
  - [x] `async stats() -> dict[str, int]`
  - [x] `async clear() -> None`
  - [x] `async node_count() -> int`
  - [x] `async edge_count() -> int`
- [x] Write comprehensive docstrings
- [x] Write `tests/graph/test_graph.py`
- [x] Test with different engine+storage combinations
- [x] Test data synchronization between engine and storage
- [x] Test smart defaults (auto-select NetworkX)
- [x] ✅ Run tests → all 183 tests passing (28 new Graph tests)
- [x] **Refactoring:** Removed `find_path()` from Storage layer (algorithms belong in Engine)
  - Eliminated 278 lines of duplicate BFS code across MemoryStorage, FileStorage, SQLStorage
  - Removed 7 duplicate tests
  - Clarified separation: Engine = algorithms, Storage = data persistence

**Deliverables:**
- `graph/graph.py` (428 lines)
- `graph/__init__.py` (37 lines - updated exports)
- `tests/graph/test_graph.py` (494 lines)
- **Total: 959 lines (post-refactoring)**

---

## 📋 Phase 5: Bundled Graph Subclasses (2-3h)
**Status:** ⬜ NOT STARTED  
**Goal:** Neo4jGraph and FalkorDBGraph (engine+storage locked together)

**Neo4jGraph Implementation:**
- [ ] Install dependency: `uv add neo4j`
- [ ] Create `graph/neo4j_graph.py`
- [ ] Write `Neo4jGraph` (inherits Graph, bundles engine+storage)
- [ ] Constructor: `__init__(uri: str, auth: tuple[str, str])`
- [ ] Use `AsyncGraphDatabase.driver()` with Cypher queries
- [ ] Implement all Graph methods (CREATE, MATCH, shortestPath, etc.)
- [ ] Add Neo4j features: indexes, constraints, transactions

**FalkorDBGraph Implementation:**
- [ ] Install dependency: `uv add redis falkordb`
- [ ] Create `graph/falkordb_graph.py`
- [ ] Write `FalkorDBGraph` (inherits Graph, bundles engine+storage)
- [ ] Constructor: `__init__(host: str, port: int, db: int)`
- [ ] Use FalkorDB client with Redis pipeline for performance
- [ ] Implement all Graph methods (Cypher-like commands)

- [ ] Write `tests/graph/test_neo4j_graph.py`
- [ ] Test Neo4jGraph (if Neo4j container available, else skip)
- [ ] Test Cypher queries
- [ ] Test path finding
- [ ] Write `tests/graph/test_falkordb_graph.py`
- [ ] Test FalkorDBGraph (if Redis available, else skip)
- [ ] Test bundled backend constraints (can't swap components)
- [ ] ✅ Run tests → all passing

**Deliverables:**
- `graph/neo4j_graph.py` (~400 lines)
- `graph/falkordb_graph.py` (~350 lines)
- `tests/graph/test_neo4j_graph.py` (~300 lines)
- `tests/graph/test_falkordb_graph.py` (~250 lines)

---

## 📋 Phase 6: Query System (2-3h)
**Status:** ✅ COMPLETE  
**Goal:** Pattern matching and complex queries

- [x] Create `graph/query.py`
- [x] Implement `QueryPattern` and `QueryResult` dataclasses
- [x] Implement `parse_pattern()` for dict-based pattern parsing
  - [x] Entity filter patterns: `{"type": "Person", "age": 30}`
  - [x] Single-hop patterns: `{"source": {...}, "relation": "...", "target": {...}}`
  - [x] Multi-hop patterns: `{"path": [{...}, {...}]}`
- [x] Implement `execute_pattern()` with query execution
  - [x] `_execute_entity_filter()` for entity queries
  - [x] `_execute_single_hop()` for relationship queries
  - [x] `_execute_multi_hop()` for path queries
- [x] Query optimization:
  - [x] Filter early (most restrictive first)
  - [x] Minimize graph traversal
  - [x] Reuse existing Graph methods (find_entities, get_relationships)
- [x] Add `Graph.match(pattern)` method
- [x] Support wildcards (`"?"` for any value)
- [x] Direct methods (already implemented in Graph):
  - [x] `find_entities(type=..., attributes=...)`
  - [x] `get_relationships(source=..., relation=..., target=...)`
  - [x] `find_path(start, end)`
- [x] Write `tests/graph/test_query.py`
- [x] Test pattern parsing (5 tests)
- [x] Test entity filtering (5 tests)
- [x] Test single-hop matching (6 tests)
- [x] Test multi-hop paths (4 tests)
- [x] Test wildcards (2 tests)
- [x] Test Graph.match() integration (3 tests)
- [x] Test edge cases (3 tests)
- [x] ✅ Run tests → all 211 tests passing (28 new query tests)

**Deliverables:**
- `graph/query.py` (396 lines)
- `graph/graph.py` (updated with match() method)
- `graph/__init__.py` (updated exports)
- `tests/graph/test_query.py` (430 lines)
- **Total: ~850 lines**

---

## 📋 Phase 7: Visualization (3-4h)
**Status:** ⬜ NOT STARTED  
**Goal:** Internal graph visualization

- [ ] Create `graph/visualization/` directory
- [ ] Create `graph/visualization/renderer.py`
- [ ] Write `entity_to_node(entity: Entity) -> str` (Mermaid syntax)
- [ ] Write `relationship_to_edge(rel: Relationship) -> str` (Mermaid syntax)
- [ ] Write `to_mermaid(entities, relationships, **opts) -> str`
- [ ] Write `to_graphviz(entities, relationships) -> str` (DOT format)
- [ ] Write `to_graphml(entities, relationships) -> str` (GraphML format)
- [ ] Support options:
  - [ ] `direction="LR"` or `"TB"` (Mermaid)
  - [ ] `group_by_type=True` (subgraphs)
  - [ ] `style="default"` (color schemes)
- [ ] Create `graph/visualization/styles.py`
- [ ] Write style classes (colors, shapes by entity type)
- [ ] Create `graph/visualization/__init__.py`
- [ ] Integrate visualization methods in `Graph` class:
  - [ ] `async show()` - Display in notebook
  - [ ] `async save_diagram(path, format="mermaid")` - Save to file
  - [ ] `async to_mermaid(**opts) -> str` - Get Mermaid code
  - [ ] `async to_graphviz() -> str` - Get DOT format
- [ ] Write `tests/graph/test_visualization.py`
- [ ] Test Mermaid code generation
- [ ] Test Graphviz/GraphML generation
- [ ] Test entity grouping
- [ ] Test style application
- [ ] ✅ Run tests → all passing

**Deliverables:**
- `graph/visualization/renderer.py` (~250 lines)
- `graph/visualization/styles.py` (~100 lines)
- `tests/graph/test_visualization.py` (~200 lines)

---

## 🏛️ Architecture & Design Decisions

### Engine vs Storage Separation

**Engine Layer** (How graph operations work):
- **NativeEngine** - Pure Python, zero dependencies, basic algorithms
- **NetworkXEngine** - 100+ graph algorithms, rich features
- **Future** RustworkxEngine (10-100x faster), CypherEngine (Cypher queries)

**Storage Layer** (Where data lives):
- **MemoryStorage** - In-memory, transient, fast
- **FileStorage** - JSON/pickle, simple persistence
- **SQLStorage** - Any SQL database (PostgreSQL, MySQL, SQLite) via SQLAlchemy v2 ORM with DRY CRUD
- **Future:** Neo4jStorage (for Neo4jGraph only)

**Key Insight:** SQL is storage, NOT a bundled backend!
- ✅ Valid: `Graph(engine=NetworkXEngine(), storage=SQLStorage("postgresql://..."))`
- Gives you: NetworkX algorithms + SQL persistence

### Bundled Backends

Only for native graph databases where engine+storage can't be separated:
- **Neo4jGraph** - Neo4j engine + Neo4j storage (locked together)
- **FalkorDBGraph** - FalkorDB engine + Redis storage (locked together)

### Query API Design

**Direct methods** (80% of queries - simple, Pythonic):
```python
people = await graph.find_entities(type="Person")
rels = await graph.find_relationships(source="alice", relation="works_at")
path = await graph.find_path("alice", "bob")
```

**Pattern matching** (20% of queries - complex, powerful):
```python
results = await graph.match({
    "path": [
        {"source": {"type": "Person"}, "relation": "works_at"},
        {"relation": "manages", "target": {"id": "project_x"}}
    ]
})
```

**No chaining, no string queries:**
- ❌ `graph.query("? -works_at-> Acme")` (confusing syntax)
- ❌ `graph.find().where().filter()` (fluent API)

---

## 📊 Progress Summary

| Phase | Status | Time | Deliverable |
|-------|--------|------|-------------|
| Phase 1: Models | ✅ | 1-2h | `models.py` (209 lines) ✅ |
| Phase 2: Storage Protocol | ✅ | 1h | `storage/base.py` (501 lines) ✅ |
| Phase 3A: Engines | ✅ | 3-4h | `engines/` (814 lines) ✅ |
| Phase 3B: Storage Impls | ✅ | 3-4h | `storage/` (1,475 lines) ✅ |
| Phase 4: Base Graph | ✅ | 2-3h | `graph.py` (428 lines) ✅ |
| Phase 5: Bundled Graphs | ⬜ SKIPPED | - | `neo4j_graph.py`, `falkordb_graph.py` |
| Phase 6: Query System | ✅ | 2-3h | `query.py` (396 lines) ✅ |
| Phase 7: Visualization | ⬜ | 3-4h | `visualization/` (~350 lines) |
| Phase 8: Docs & Polish | ⬜ | 2-3h | README, examples, mkdocs |
| **TOTAL** | **~70%** | **12-19h** | **~3,800+ lines + 211 tests** |

---

## ✅ Success Checklist

### Code Quality
- [x] All tests passing (123/123 tests)
- [ ] >90% test coverage
- [x] Zero import errors
- [ ] Ruff linting clean
- [x] Complete type hints (Python 3.13+)
- [x] Zero code copied from old implementation

### API Design
- [ ] Simple import: `from cogent.graph import Graph`
- [ ] Smart defaults: `Graph()` works out of the box
- [ ] Explicit composition: `Graph(engine=..., storage=...)`
- [ ] Bundled backends: `Neo4jGraph(uri, auth)`, `FalkorDBGraph(host, port)`
- [ ] Direct query methods: `find_entities()`, `find_relationships()`, `find_path()`
- [ ] Pattern matching: `match(pattern_dict)`
- [ ] Visualization: `show()`, `save_diagram()`, `to_mermaid()`

### Documentation
- [ ] Comprehensive README with architecture overview
- [ ] API docs complete
- [ ] Usage examples in `/examples/graph/`
- [ ] mkdocs page created
- [ ] Migration guide drafted (for future capability migration)

---

## 🎯 Current Focus

**Phases 1-3A ✅ COMPLETE!**
- ✅ Created `graph/models.py` (209 lines)
- ✅ Created `graph/storage.py` (263 lines)
- ✅ Created `graph/engines/` (base.py, simple.py, networkx.py) (814 lines)
- ✅ Wrote comprehensive tests (1,244 lines)
- ✅ All 123 tests passing (100% pass rate)
- ✅ NetworkX is a required dependency (not optional)
- ✅ NativeEngine is the fallback/alternative

**Overall Progress:**
- ✅ Phase 1: Models (535 lines, 30 tests)
- ✅ Phase 2: Async Storage Protocol (670 lines, 22 tests)
- ✅ Phase 3A: Engine Implementations (1,333 lines, 71 tests)
- **Total: 2,538 lines code + 123 tests passing**

**Architecture Principles Applied:**
- ✅ **Modularity:** Engine and storage completely separated
- ✅ **Scalability:** Async-first API, supports massive graphs (Neo4j)
- ✅ **Extendability:** Easy to add engines (Rustworkx) and storage (Redis)
- ✅ **Simplicity:** Smart defaults, works out of the box

**Next Phase:**
**Phase 3B: Storage Implementations** (3-4h)
1. Refactor `graph/storage.py` → `graph/storage/base.py`
2. Create `graph/storage/` directory
3. Write `storage/memory.py` - MemoryStorage (transient)
4. Write `storage/file.py` - FileStorage (JSON/pickle)
5. Write `storage/sql.py` - SQLStorage (SQLAlchemy v2 ORM + DRY CRUD)
6. Write comprehensive tests for all storage implementations

**Status:** ✅ Phase 3A complete - Ready for Phase 3B implementation

