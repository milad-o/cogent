"""Tests for graph engines."""

import pytest
from cogent.graph.engines import (
    Engine,
    NativeEngine,
    NetworkXEngine,
)


# --- Fixtures ---


@pytest.fixture
def native_engine() -> NativeEngine:
    """Create a fresh NativeEngine."""
    return NativeEngine()


@pytest.fixture
def networkx_engine() -> NetworkXEngine:
    """Create a fresh NetworkXEngine."""
    return NetworkXEngine()


def pytest_generate_tests(metafunc):
    """Dynamically parametrize engine fixture."""
    if "engine" in metafunc.fixturenames:
        metafunc.parametrize("engine", ["native", "networkx"], indirect=True)


@pytest.fixture
def engine(request: pytest.FixtureRequest) -> Engine:
    """Parametrized fixture for all engines."""
    if request.param == "native":
        return NativeEngine()
    if request.param == "networkx":
        return NetworkXEngine()
    raise ValueError(f"Unknown engine: {request.param}")


# --- Protocol Conformance Tests ---


def test_native_engine_implements_protocol(native_engine: NativeEngine) -> None:
    """NativeEngine implements Engine protocol."""
    engine: Engine = native_engine
    assert engine is not None


def test_networkx_engine_implements_protocol(networkx_engine: NetworkXEngine) -> None:
    """NetworkXEngine implements Engine protocol."""
    engine: Engine = networkx_engine
    assert engine is not None


# --- Node Operation Tests ---


@pytest.mark.asyncio
async def test_add_node(engine: Engine) -> None:
    """Add a node with attributes."""
    await engine.add_node("alice", name="Alice", age=30)

    node = await engine.get_node("alice")
    assert node is not None
    assert node["name"] == "Alice"
    assert node["age"] == 30


@pytest.mark.asyncio
async def test_add_node_no_attributes(engine: Engine) -> None:
    """Add a node without attributes."""
    await engine.add_node("bob")

    node = await engine.get_node("bob")
    assert node is not None
    assert len(node) == 0


@pytest.mark.asyncio
async def test_remove_node(engine: Engine) -> None:
    """Remove a node."""
    await engine.add_node("alice")
    assert await engine.has_node("alice")

    removed = await engine.remove_node("alice")
    assert removed is True
    assert not await engine.has_node("alice")


@pytest.mark.asyncio
async def test_remove_nonexistent_node(engine: Engine) -> None:
    """Remove a non-existent node returns False."""
    removed = await engine.remove_node("nonexistent")
    assert removed is False


@pytest.mark.asyncio
async def test_remove_node_removes_edges(engine: Engine) -> None:
    """Removing a node also removes its edges."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")

    await engine.remove_node("bob")

    # Check edges are gone
    edges = await engine.get_edges()
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_get_node_nonexistent(engine: Engine) -> None:
    """Get non-existent node returns None."""
    node = await engine.get_node("nonexistent")
    assert node is None


@pytest.mark.asyncio
async def test_has_node(engine: Engine) -> None:
    """Check if node exists."""
    assert not await engine.has_node("alice")

    await engine.add_node("alice")
    assert await engine.has_node("alice")


@pytest.mark.asyncio
async def test_node_count(engine: Engine) -> None:
    """Count nodes."""
    assert await engine.node_count() == 0

    await engine.add_node("alice")
    await engine.add_node("bob")
    assert await engine.node_count() == 2


@pytest.mark.asyncio
async def test_get_neighbors_outgoing(engine: Engine) -> None:
    """Get outgoing neighbors."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("alice", "charlie")

    neighbors = await engine.get_neighbors("alice", "outgoing")
    assert set(neighbors) == {"bob", "charlie"}


@pytest.mark.asyncio
async def test_get_neighbors_incoming(engine: Engine) -> None:
    """Get incoming neighbors."""
    await engine.add_edge("bob", "alice")
    await engine.add_edge("charlie", "alice")

    neighbors = await engine.get_neighbors("alice", "incoming")
    assert set(neighbors) == {"bob", "charlie"}


@pytest.mark.asyncio
async def test_get_neighbors_both(engine: Engine) -> None:
    """Get all neighbors (both directions)."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("charlie", "alice")

    neighbors = await engine.get_neighbors("alice", "both")
    assert set(neighbors) == {"bob", "charlie"}


@pytest.mark.asyncio
async def test_get_neighbors_nonexistent(engine: Engine) -> None:
    """Get neighbors of non-existent node returns empty list."""
    neighbors = await engine.get_neighbors("nonexistent")
    assert neighbors == []


# --- Edge Operation Tests ---


@pytest.mark.asyncio
async def test_add_edge(engine: Engine) -> None:
    """Add an edge with attributes."""
    await engine.add_edge("alice", "bob", relation="knows", since=2020)

    edges = await engine.get_edges()
    assert len(edges) == 1
    src, tgt, attrs = edges[0]
    assert src == "alice"
    assert tgt == "bob"
    assert attrs["relation"] == "knows"
    assert attrs["since"] == 2020


@pytest.mark.asyncio
async def test_add_edge_creates_nodes(engine: Engine) -> None:
    """Adding edge creates nodes if they don't exist."""
    await engine.add_edge("alice", "bob")

    assert await engine.has_node("alice")
    assert await engine.has_node("bob")


@pytest.mark.asyncio
async def test_remove_edge(engine: Engine) -> None:
    """Remove an edge."""
    await engine.add_edge("alice", "bob")
    assert await engine.edge_count() == 1

    removed = await engine.remove_edge("alice", "bob")
    assert removed is True
    assert await engine.edge_count() == 0


@pytest.mark.asyncio
async def test_remove_nonexistent_edge(engine: Engine) -> None:
    """Remove non-existent edge returns False."""
    removed = await engine.remove_edge("alice", "bob")
    assert removed is False


@pytest.mark.asyncio
async def test_get_edges_all(engine: Engine) -> None:
    """Get all edges."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")

    edges = await engine.get_edges()
    assert len(edges) == 2


@pytest.mark.asyncio
async def test_get_edges_by_source(engine: Engine) -> None:
    """Get edges from specific source."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("alice", "charlie")
    await engine.add_edge("bob", "charlie")

    edges = await engine.get_edges(source_id="alice")
    assert len(edges) == 2
    for src, _, _ in edges:
        assert src == "alice"


@pytest.mark.asyncio
async def test_get_edges_by_target(engine: Engine) -> None:
    """Get edges to specific target."""
    await engine.add_edge("alice", "charlie")
    await engine.add_edge("bob", "charlie")

    edges = await engine.get_edges(target_id="charlie")
    assert len(edges) == 2
    for _, tgt, _ in edges:
        assert tgt == "charlie"


@pytest.mark.asyncio
async def test_get_edges_by_source_and_target(engine: Engine) -> None:
    """Get edges with both source and target filter."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("alice", "charlie")

    edges = await engine.get_edges(source_id="alice", target_id="bob")
    assert len(edges) == 1
    src, tgt, _ = edges[0]
    assert src == "alice"
    assert tgt == "bob"


@pytest.mark.asyncio
async def test_edge_count(engine: Engine) -> None:
    """Count edges."""
    assert await engine.edge_count() == 0

    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")
    assert await engine.edge_count() == 2


# --- Graph Operation Tests ---


@pytest.mark.asyncio
async def test_find_path_direct(engine: Engine) -> None:
    """Find direct path."""
    await engine.add_edge("alice", "bob")

    path = await engine.find_path("alice", "bob")
    assert path == ["alice", "bob"]


@pytest.mark.asyncio
async def test_find_path_indirect(engine: Engine) -> None:
    """Find indirect path."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")

    path = await engine.find_path("alice", "charlie")
    assert path == ["alice", "bob", "charlie"]


@pytest.mark.asyncio
async def test_find_path_no_path(engine: Engine) -> None:
    """No path returns None."""
    await engine.add_node("alice")
    await engine.add_node("bob")

    path = await engine.find_path("alice", "bob")
    assert path is None


@pytest.mark.asyncio
async def test_find_path_same_node(engine: Engine) -> None:
    """Path to same node returns single-element list."""
    await engine.add_node("alice")

    path = await engine.find_path("alice", "alice")
    assert path == ["alice"]


@pytest.mark.asyncio
async def test_find_path_nonexistent_nodes(engine: Engine) -> None:
    """Path with non-existent nodes returns None."""
    path = await engine.find_path("alice", "bob")
    assert path is None


@pytest.mark.asyncio
async def test_find_path_max_depth(engine: Engine) -> None:
    """Find path respects max_depth."""
    await engine.add_edge("a", "b")
    await engine.add_edge("b", "c")
    await engine.add_edge("c", "d")

    # Path exists but exceeds max_depth
    path = await engine.find_path("a", "d", max_depth=2)
    assert path is None

    # Path within max_depth
    path = await engine.find_path("a", "c", max_depth=2)
    assert path == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_connected_components_single(engine: Engine) -> None:
    """Single connected component."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")

    components = await engine.connected_components()
    assert len(components) == 1
    assert components[0] == {"alice", "bob", "charlie"}


@pytest.mark.asyncio
async def test_connected_components_multiple(engine: Engine) -> None:
    """Multiple connected components."""
    # Component 1
    await engine.add_edge("alice", "bob")

    # Component 2
    await engine.add_edge("charlie", "dave")

    components = await engine.connected_components()
    assert len(components) == 2

    component_sets = [set(c) for c in components]
    assert {"alice", "bob"} in component_sets
    assert {"charlie", "dave"} in component_sets


@pytest.mark.asyncio
async def test_connected_components_isolated_nodes(engine: Engine) -> None:
    """Isolated nodes form separate components."""
    await engine.add_node("alice")
    await engine.add_node("bob")

    components = await engine.connected_components()
    assert len(components) == 2


@pytest.mark.asyncio
async def test_get_subgraph(engine: Engine) -> None:
    """Extract subgraph."""
    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")
    await engine.add_edge("charlie", "dave")

    subgraph = await engine.get_subgraph({"alice", "bob"})

    # Check nodes
    assert await subgraph.node_count() == 2
    assert await subgraph.has_node("alice")
    assert await subgraph.has_node("bob")

    # Check edges (only edges within subgraph)
    edges = await subgraph.get_edges()
    assert len(edges) == 1
    src, tgt, _ = edges[0]
    assert src == "alice"
    assert tgt == "bob"


@pytest.mark.asyncio
async def test_clear(engine: Engine) -> None:
    """Clear removes all nodes and edges."""
    await engine.add_edge("alice", "bob", relation="knows")
    await engine.add_node("charlie", age=25)

    await engine.clear()

    assert await engine.node_count() == 0
    assert await engine.edge_count() == 0


# --- NetworkX-Specific Tests ---


@pytest.mark.asyncio
async def test_networkx_betweenness_centrality() -> None:
    """NetworkX betweenness centrality."""
    engine = NetworkXEngine()

    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")

    centrality = engine.betweenness_centrality()
    assert "alice" in centrality
    assert "bob" in centrality
    assert "charlie" in centrality

    # Bob is in the middle, should have highest centrality
    assert centrality["bob"] > centrality["alice"]
    assert centrality["bob"] > centrality["charlie"]


@pytest.mark.asyncio
async def test_networkx_pagerank() -> None:
    """NetworkX PageRank."""
    engine = NetworkXEngine()

    await engine.add_edge("alice", "bob")
    await engine.add_edge("bob", "charlie")
    await engine.add_edge("charlie", "bob")  # Bob gets more incoming edges

    scores = engine.pagerank()
    assert "alice" in scores
    assert "bob" in scores
    assert "charlie" in scores


@pytest.mark.asyncio
async def test_networkx_all_simple_paths() -> None:
    """NetworkX all simple paths."""
    engine = NetworkXEngine()

    # Create diamond graph
    await engine.add_edge("a", "b")
    await engine.add_edge("a", "c")
    await engine.add_edge("b", "d")
    await engine.add_edge("c", "d")

    paths = engine.all_simple_paths("a", "d")
    assert len(paths) == 2
    assert ["a", "b", "d"] in paths
    assert ["a", "c", "d"] in paths


@pytest.mark.asyncio
async def test_networkx_is_directed_acyclic() -> None:
    """NetworkX DAG detection."""
    engine = NetworkXEngine()

    # DAG
    await engine.add_edge("a", "b")
    await engine.add_edge("b", "c")
    assert engine.is_directed_acyclic() is True

    # Add cycle
    await engine.add_edge("c", "a")
    assert engine.is_directed_acyclic() is False


@pytest.mark.asyncio
async def test_networkx_topological_sort() -> None:
    """NetworkX topological sort."""
    engine = NetworkXEngine()

    await engine.add_edge("a", "b")
    await engine.add_edge("b", "c")
    await engine.add_edge("a", "c")

    order = engine.topological_sort()
    assert len(order) == 3

    # 'a' should come before 'b' and 'c'
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    # 'b' should come before 'c'
    assert order.index("b") < order.index("c")
