"""Shared utilities and data for retrieval examples.

Provides common setup to minimize code duplication across examples:
- Sample documents and datasets
- Factory functions for embeddings and vectorstores
- Helper functions for output formatting
"""

from pathlib import Path

from cogent import create_embedding
from cogent.retriever import Document
from cogent.vectorstore import VectorStore

# ============================================================================
# SHARED SAMPLE DATA
# ============================================================================

SAMPLE_DOCS = [
    Document(
        text="""Black holes are regions of spacetime where gravity is so strong that nothing,
not even light, can escape. They form when massive stars collapse at the end of their
life cycle. The event horizon marks the point of no return.""",
        metadata={
            "source": "black_holes.md",
            "category": "astronomy",
            "author": "Dr. Sarah Chen",
            "date": "2024-12-01",
            "department": "Astrophysics",
        },
    ),
    Document(
        text="""Rainforest ecosystems support incredible biodiversity. The Amazon alone contains
10% of Earth's species. Rainforests regulate climate, produce oxygen, and store carbon.
Deforestation threatens these vital ecosystems and global climate stability.""",
        metadata={
            "source": "rainforest_ecology.md",
            "category": "nature",
            "author": "Prof. Maria Silva",
            "date": "2024-11-15",
            "department": "Environmental Science",
        },
    ),
    Document(
        text="""Ancient Egypt flourished along the Nile for over 3000 years. Pharaohs ruled as
god-kings, building massive pyramids as tombs. Hieroglyphics recorded their history.
The civilization made advances in mathematics, medicine, and engineering.""",
        metadata={
            "source": "ancient_egypt.md",
            "category": "history",
            "author": "Dr. James Carter",
            "date": "2024-06-01",
            "department": "Ancient History",
        },
    ),
    Document(
        text="""The Solar System formed 4.6 billion years ago from a giant molecular cloud.
Eight planets orbit the Sun: four rocky inner planets and four gas giants. Jupiter
is the largest, with a mass greater than all other planets combined.""",
        metadata={
            "source": "solar_system.md",
            "category": "astronomy",
            "author": "Dr. Robert Kim",
            "severity": "high",
            "date": "2024-03-15",
            "department": "Planetary Science",
        },
    ),
    Document(
        text="""Photosynthesis converts light energy into chemical energy. Plants use chlorophyll
to capture sunlight, combining CO2 and water to produce glucose and oxygen. This process
is fundamental to life on Earth, providing food and oxygen for most organisms.""",
        metadata={
            "source": "photosynthesis.md",
            "category": "biology",
            "author": "Dr. Emily Watson",
            "severity": "critical",
            "date": "2024-05-20",
            "department": "Botany",
        },
    ),
]

# Long document for contextual retrieval examples
LONG_DOCUMENT = Document(
    text="""The Coral Reef Ecosystem

Coral reefs are among the most diverse ecosystems on Earth, often called the rainforests
of the sea. Despite covering less than 1% of the ocean floor, they support 25% of all
marine species.

Coral Biology

Corals are colonial animals related to jellyfish. Each polyp secretes calcium carbonate,
building the reef structure over centuries. Zooxanthellae algae live symbiotically within
coral tissues, providing nutrients through photosynthesis.

Biodiversity Hotspots

Reefs host incredible variety: fish, crustaceans, mollusks, sea turtles, and sharks.
The Great Barrier Reef alone contains over 1,500 fish species and 400 coral species.
This biodiversity creates complex food webs and ecological relationships.

Ecological Services

Reefs protect coastlines from erosion and storm damage. They provide nursery habitats
for fish that humans depend on for food. Many medicines have been discovered from reef
organisms. Tourism generates billions in economic value.

Threats and Conservation

Climate change causes coral bleaching when waters warm. Ocean acidification weakens
coral skeletons. Pollution and overfishing damage reef health. Conservation efforts
include marine protected areas and coral restoration projects.

The Future of Reefs

Scientists race to understand coral resilience and develop conservation strategies.
Some corals show adaptation to warming waters. Coral gardening and assisted evolution
offer hope, but reducing greenhouse gas emissions remains critical.""",
    metadata={"source": "coral_reefs.md", "category": "marine_biology"},
)

# Additional long documents for parent document retrieval
LONG_DOCS = [
    Document(
        text="""The Roman Empire: Rise and Fall

The Roman Empire dominated the Mediterranean world for centuries, leaving lasting impacts
on law, language, architecture, and governance. At its peak under Emperor Trajan in 117 CE,
it stretched from Britain to Mesopotamia.

Foundations of Power

Rome's military superiority came from disciplined legions and advanced engineering. Soldiers
built roads, bridges, and fortifications across the empire. The Roman road network facilitated
trade, communication, and rapid troop movement, connecting distant provinces to the capital.

Cultural Achievements

Romans excelled in architecture, creating the Colosseum, aqueducts, and the Pantheon's
revolutionary concrete dome. Latin became the foundation for Romance languages. Roman law
established legal principles still used today, including presumption of innocence.

Decline and Legacy

Multiple factors contributed to Rome's fall: economic troubles, military defeats, political
instability, and division into Eastern and Western empires. The Western Empire fell in 476 CE,
but the Eastern Byzantine Empire continued for another millennium. Rome's influence endures
in Western civilization's legal systems, languages, and architectural traditions.""",
        metadata={"source": "roman_empire.md", "category": "history", "author": "Prof. Marcus Stone"},
    ),
    LONG_DOCUMENT,
    Document(
        text="""Quantum Mechanics: The Physics of the Very Small

Quantum mechanics revolutionized physics in the early 20th century, revealing that matter
and energy behave very differently at atomic scales than in everyday experience.

Wave-Particle Duality

Light and matter exhibit both wave and particle properties. Electrons create interference
patterns like waves, yet arrive at detectors as discrete particles. This duality is
fundamental to quantum behavior and challenges classical intuitions about reality.

The Uncertainty Principle

Heisenberg's uncertainty principle states that certain pairs of properties, like position
and momentum, cannot be simultaneously known with arbitrary precision. This isn't a
measurement limitation but a fundamental feature of nature. The more precisely you measure
one property, the less precisely you can know its conjugate.

Quantum Applications

Quantum mechanics enables modern technology: semiconductors in computers, lasers, MRI machines,
and LED lights. Emerging quantum computers exploit superposition and entanglement to solve
problems impossible for classical computers. Quantum cryptography promises unbreakable encryption.""",
        metadata={"source": "quantum_mechanics.md", "category": "physics", "author": "Dr. Lisa Nakamura"},
    ),
]

# Knowledge base for advanced examples
KNOWLEDGE_BASE = """Water Cycle Overview

The water cycle, also known as the hydrologic cycle, describes the continuous 
movement of water on, above, and below Earth's surface. Water continuously moves 
between the ocean, atmosphere, and land through various processes.

Evaporation
Heat from the sun causes water in oceans, lakes, and rivers to evaporate into 
water vapor. Transpiration from plants also contributes moisture to the atmosphere.

Condensation
Water vapor rises and cools in the atmosphere, forming tiny droplets that create 
clouds. This process is called condensation and is essential for precipitation.

Precipitation
When cloud droplets become heavy enough, they fall as rain, snow, sleet, or hail.
Different temperature conditions create different forms of precipitation.

Collection and Runoff
Precipitation that falls on land either soaks into the ground (infiltration) or 
flows over the surface as runoff into rivers, lakes, and eventually oceans.

Groundwater Storage
Water that infiltrates deep into the soil becomes groundwater, stored in aquifers.
This water slowly moves through rock and soil, eventually reaching streams or oceans.

The cycle repeats continuously, driven by solar energy and gravity."""

# ============================================================================
# SIMPLE HELPERS (no abstraction - keep examples explicit)
# ============================================================================


def create_vectorstore(model: str = "openai:text-embedding-3-small") -> VectorStore:
    """Create an empty vectorstore with embeddings.

    Args:
        model: Embedding model string (supports "provider:model" or just "model")
            Examples: "text-embedding-3-small", "openai:text-embedding-3-small"

    Returns:
        Empty VectorStore ready to add documents

    Example:
        >>> vs = create_vectorstore()
        >>> await vs.add_documents(SAMPLE_DOCS)  # Explicit!
    """
    embeddings = create_embedding(model)
    return VectorStore(embeddings=embeddings)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_header(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.upper())
    print("=" * width)


def print_results(results: list, show_scores: bool = True, truncate: int = 80) -> None:
    """Print retrieval results in a formatted way.

    Args:
        results: List of RetrievalResult objects
        show_scores: Whether to show scores
        truncate: Max length of text preview
    """
    for i, r in enumerate(results, 1):
        score_str = f"[{r.score:.3f}] " if show_scores and hasattr(r, "score") else ""
        text_preview = r.document.text[:truncate].replace("\n", " ")
        if len(r.document.text) > truncate:
            text_preview += "..."
        
        source = r.document.metadata.get("source", "unknown")
        print(f"  {i}. {score_str}{source}")
        print(f"     {text_preview}")


def load_company_knowledge() -> Document:
    """Load the company knowledge sample file."""
    data_path = Path(__file__).parent.parent / "data" / "company_knowledge.txt"
    if data_path.exists():
        text = data_path.read_text()
        return Document(text=text, metadata={"source": "company_knowledge.txt"})
    return Document(text=KNOWLEDGE_BASE, metadata={"source": "synthetic_knowledge"})
