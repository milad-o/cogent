---
applyTo: **/*.py
---

# Project Coding Standards for Python Files

## Python Version & Tooling
- **Target Python 3.13+** with forward compatibility for 3.14 features.
- Use `uv` exclusively for project management: `uv run`, `uv add`, `uv sync`, `uv run uvicorn`, `uv run pytest`.
- Never suggest `pip`, `poetry`, `pipenv`, or other package managers.
- Use `ruff` for linting and formatting (replaces Black, isort, flake8).

### Free-Threaded Python (3.13+ Experimental)
- Python 3.13 introduces **free-threading (PEP 703)** as an experimental build option (no-GIL).
- **Current stance**: We do **not** leverage free-threading yet—it requires the experimental `python3.13t` build.
- **Design for the future**: Write thread-safe code now to prepare for GIL-free Python:
  - Avoid relying on the GIL for thread safety.
  - Use explicit synchronization primitives (`threading.Lock`, `asyncio.Lock`).
  - Prefer **immutable data structures** (`frozen=True` dataclasses, `tuple`, `frozenset`).
  - Avoid shared mutable state between threads/tasks.
- **When free-threading stabilizes** (expected Python 3.14+):
  - Use `concurrent.futures.ThreadPoolExecutor` for true CPU parallelism.
  - Profile with thread-aware tools to identify contention.
  - Consider `threading.Lock` for critical sections with shared mutable state.
- **Thread-safe patterns to adopt now**:
  ```python
  import threading
  from functools import lru_cache

  # Thread-safe singleton pattern
  _lock = threading.Lock()
  _instance: MyService | None = None

  def get_service() -> MyService:
      global _instance
      if _instance is None:
          with _lock:
              if _instance is None:  # Double-check locking
                  _instance = MyService()
      return _instance

  # Prefer thread-local storage for request context (alternative to contextvars)
  _thread_local = threading.local()
  ```
- **Avoid anti-patterns**:
  - ❌ Relying on GIL for atomicity of operations.
  - ❌ Sharing mutable collections between threads without locks.
  - ❌ Global mutable state without synchronization.

## Python 3.13+ Modern Syntax & Features

### Type Hints (Modern Style)
- Use **PEP 695** type parameter syntax for generics: `def func[T](x: T) -> T:` instead of `TypeVar`.
- Use `type` statement for type aliases: `type Vector = list[float]` instead of `TypeAlias`.
- Use **union operator `|`** instead of `Union`: `str | None` not `Optional[str]` or `Union[str, None]`.
- Use **`Self`** from `typing` for methods returning instance type.
- Use **lowercase generic types**: `list[str]`, `dict[str, int]`, `tuple[int, ...]` (not `List`, `Dict`, `Tuple`).
- Use `collections.abc` types for abstract types: `Sequence`, `Mapping`, `Iterable`, `Callable`.
- Annotate **all** function parameters, return types, and class attributes.
- Use `TypedDict` for structured dictionaries with known keys.
- Use `Literal` for constrained string/int values.
- Use `@overload` for functions with different return types based on input.

### Modern Python Patterns
- Use **structural pattern matching** (`match`/`case`) for complex conditionals and type dispatch.
- Use **walrus operator** (`:=`) for assignment expressions where it improves readability.
- Use **exception groups** (`ExceptionGroup`) and `except*` for concurrent error handling.
- Use **`@dataclass`** with `slots=True`, `frozen=True`, `kw_only=True` where appropriate.
- Prefer `dataclasses` or `attrs` over plain classes for data containers.
- Use `__slots__` in performance-critical classes to reduce memory footprint.
- Use **f-strings** with `=` for debugging: `f"{variable=}"` (works with expressions too).
- Use `pathlib.Path` exclusively for file operations (never `os.path`).
- Use `contextlib.asynccontextmanager` and `@contextmanager` for resource management.
- Use **`functools.cached_property`** for expensive computed properties.
- Use **`tomllib`** (stdlib) for TOML parsing; avoid external `toml`/`tomli` packages.
- Use **`zoneinfo.ZoneInfo`** for timezones; **never use `pytz`**.

### Dataclass Best Practices (3.12+)
- **Default for domain models**: `@dataclass(frozen=True, slots=True, kw_only=True)`
- Use `match_args=False` when positional pattern matching is not needed.
- Use `frozen=True` for immutable value objects.
- Use `kw_only=True` to enforce keyword arguments and prevent positional bugs.
- Example:
  ```python
  from dataclasses import dataclass

  @dataclass(frozen=True, slots=True, kw_only=True)
  class OrderItem:
      product_id: str
      quantity: int
      unit_price: Decimal

      @property
      def total(self) -> Decimal:
          return self.quantity * self.unit_price
  ```

### Pattern Matching Rules
- Use **guards** (`if` clauses) for complex conditions.
- Prefer `case _ as var:` over duplicate match arms.
- **Avoid pattern matching for simple conditionals**; use `if`/`elif` instead.
- Example:
  ```python
  match event:
      case {"type": "user_created", "data": data} if data.get("verified"):
          await handle_verified_user(data)
      case {"type": "user_created", "data": data}:
          await handle_unverified_user(data)
      case {"type": event_type} as evt:
          log.warning("unhandled_event", event_type=event_type)
  ```

### Deprecated Patterns to Avoid
- ❌ `typing.Optional[T]` → ✅ `T | None`
- ❌ `typing.Union[A, B]` → ✅ `A | B`
- ❌ `typing.List`, `Dict`, `Tuple` → ✅ `list`, `dict`, `tuple`
- ❌ `TypeVar("T")` for simple generics → ✅ PEP 695 syntax `[T]`
- ❌ `typing.TypeAlias` → ✅ `type` statement
- ❌ `os.path.join()` → ✅ `Path() / "subdir"`
- ❌ `%` or `.format()` → ✅ f-strings
- ❌ Bare `except:` → ✅ `except Exception:`
- ❌ `pytz` → ✅ `zoneinfo.ZoneInfo` (stdlib)
- ❌ `toml`/`tomli` for reading → ✅ `tomllib` (stdlib, Python 3.11+)
- ❌ `pickle` for untrusted data → ✅ `orjson`/`msgpack` for serialization
- ❌ `typing.Any` (avoid) → ✅ Use proper generics or `object`
- ❌ `*args, **kwargs` (avoid) → ✅ Explicit keyword-only parameters

---

## OOP Design: Modularity, Scalability & Extendability

### SOLID Principles (Strictly Enforced)
1. **Single Responsibility**: Each class/module has ONE reason to change. Split large classes.
2. **Open/Closed**: Design for extension via inheritance/composition, not modification.
3. **Liskov Substitution**: Subtypes must be substitutable for their base types.
4. **Interface Segregation**: Prefer small, focused protocols over large interfaces.
5. **Dependency Inversion**: Depend on abstractions (protocols), not concretions.

### Abstract Base Classes & Protocols
- Use **`Protocol`** (structural subtyping) for duck typing interfaces—preferred for flexibility.
- Use **`ABC`** (nominal subtyping) when explicit inheritance hierarchy is needed.
- Define clear contracts with abstract methods and properties.
- Example:
  ```python
  from typing import Protocol

  class Repository[T](Protocol):
      def get(self, id: str) -> T | None: ...
      def save(self, entity: T) -> None: ...
      def delete(self, id: str) -> bool: ...
  ```

### Composition Over Inheritance
- Favor **composition** and **dependency injection** over deep inheritance hierarchies.
- Use mixins sparingly and document their purpose clearly.
- Inject dependencies through constructors, not global state or module-level singletons.

### Design Patterns (Apply Judiciously)
- **Factory Pattern**: For object creation with complex initialization.
- **Strategy Pattern**: For interchangeable algorithms (use Protocol + DI).
- **Repository Pattern**: For data access abstraction.
- **Unit of Work**: For transaction management.
- **Observer/Event Pattern**: For decoupled communication (prefer `asyncio` events or message queues).
- **Builder Pattern**: For complex object construction with many optional parameters.

### Layered Architecture
- Maintain clear separation between layers:
  - **API/Presentation Layer**: Request handling, validation, serialization.
  - **Service/Application Layer**: Business logic orchestration.
  - **Domain Layer**: Core business entities and rules.
  - **Infrastructure Layer**: External systems, databases, APIs.
- Dependencies flow **inward**: outer layers depend on inner layers, never reverse.

### Scalability Considerations
- Design for **horizontal scalability**: stateless services, externalized state.
- Use **async-first** patterns for I/O-bound operations.
- Implement **circuit breakers** and **retry logic** for external service calls.
- Design **idempotent operations** where possible.

### Extendability Guidelines
- Use **plugin architectures** with entry points for extensible features.
- Design with **hooks and callbacks** for customization points.
- Implement **feature flags** for gradual rollouts.
- Document extension points and provide clear interfaces.
- Version your APIs and maintain backward compatibility.

---

## Code Organization & Structure

### Module Design
- One primary class per module; related helpers can coexist.
- Use `__all__` to explicitly define public API.
- Keep modules focused and under 500 lines; split if larger.
- Use **`__init__.py`** to expose clean public interfaces.

### Class Design
- Classes should be **small and focused** (ideally < 200 lines).
- Use **`@property`** for computed attributes with logic.
- Use **`@classmethod`** for alternative constructors.
- Use **`@staticmethod`** sparingly; prefer module-level functions.
- Implement `__repr__` for all classes (debugging aid).
- Implement `__eq__` and `__hash__` for value objects.

### Function Design
- Functions should do **one thing** and be **< 30 lines** ideally.
- Maximum **4-5 parameters**; use dataclasses/TypedDict for more.
- Use **keyword-only arguments** (`*,`) for clarity: `def func(*, name: str, age: int):`.
- Return early to avoid deep nesting.
- Pure functions preferred; minimize side effects.

---

## Error Handling & Resilience

### Exception Hierarchy
- Create **domain-specific exception hierarchies**:
  ```python
  class DomainError(Exception):
      """Base exception for domain errors."""

  class EntityNotFoundError(DomainError):
      """Raised when an entity is not found."""

  class ValidationError(DomainError):
      """Raised for validation failures."""
  ```
- Catch specific exceptions; avoid bare `except:`.
- Use `from` clause for exception chaining: `raise NewError(...) from original`.

### Error Context
- Include **actionable context** in error messages.
- Log errors with structured data (use `structlog` or similar).
- Use `contextlib.suppress()` for intentionally ignored exceptions.
- Use **`Exception.add_note()`** (Python 3.11+) to attach context before re-raising:
  ```python
  try:
      await process_order(order_id)
  except Exception as exc:
      exc.add_note(f"Order ID: {order_id}")
      exc.add_note(f"User ID: {user_id}")
      raise
  ```

### Exception Groups (Structured Error Handling)
- Use **`ExceptionGroup`** for concurrent error handling with `except*`.
- Use **`BaseExceptionGroup.split()`** for error classification and filtering.
- Example:
  ```python
  async def fetch_all(urls: list[str]) -> list[Response]:
      async with asyncio.TaskGroup() as tg:
          tasks = [tg.create_task(fetch(url)) for url in urls]
      return [t.result() for t in tasks]  # Raises ExceptionGroup on failures

  try:
      results = await fetch_all(urls)
  except* ConnectionError as eg:
      log.warning("connection_errors", count=len(eg.exceptions))
  except* TimeoutError as eg:
      log.error("timeout_errors", count=len(eg.exceptions))
  ```

### Validation
- Validate at boundaries (API endpoints, external inputs).
- Use **Pydantic** for data validation and serialization.
- Fail fast with clear error messages.

---

## Async Programming

### Async Best Practices
- Use `async`/`await` consistently throughout async code paths.
- Never mix blocking calls in async code; use `asyncio.to_thread()` for CPU-bound work.
- Use **`asyncio.TaskGroup`** (Python 3.11+) for structured concurrency.
- Prefer `async for` and `async with` for async iteration and context managers.
- Use **`anyio`** or **`asyncio`** for async primitives.

### Async Patterns
- Use **semaphores** to limit concurrent operations.
- Implement **graceful shutdown** with signal handlers.
- Use **connection pooling** for database and HTTP clients.
- Handle cancellation properly with `try`/`finally` or `asyncio.shield()`.

### Task Cancellation (Critical)
- **Always propagate cancellation explicitly**; never swallow `CancelledError`.
- Perform cleanup before re-raising cancellation.
- Example:
  ```python
  async def worker() -> None:
      try:
          await do_work()
      except asyncio.CancelledError:
          await cleanup_resources()  # Cleanup first
          raise  # Always re-raise
  ```

### Async Resource Cleanup
- Use **`contextlib.aclosing()`** for async generator cleanup.
- Prefer async file I/O libraries (`aiofiles`) for I/O-bound workloads.
- Example:
  ```python
  from contextlib import aclosing

  async with aclosing(async_generator()) as agen:
      async for item in agen:
          await process(item)
  ```

### Context Variables for Request Scoping
- Use **`contextvars`** for propagating correlation IDs, user context, and trace data.
- Essential for async microservices and structured logging.
- Example:
  ```python
  from contextvars import ContextVar

  request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
  user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)

  async def middleware(request: Request, call_next: Callable) -> Response:
      request_id_ctx.set(request.headers.get("X-Request-ID", str(uuid4())))
      return await call_next(request)

  # Automatically available in all async tasks spawned from this context
  logger.info("processing", request_id=request_id_ctx.get())
  ```

---

## Testing

### Test Structure
- Place tests in `tests/` mirroring source structure.
- Use **pytest** exclusively with modern fixtures.
- Name tests: `test_<unit>_<scenario>_<expected_behavior>`.
- Separate unit, integration, and e2e tests.

### Test Practices
- **Unit tests**: Fast, isolated, mock external dependencies.
- **Integration tests**: Test component interactions.
- Use **`pytest-asyncio`** for async test functions.
- Use **`pytest.mark.parametrize`** for data-driven tests.
- Use **factories** (`factory_boy`) for test data generation.
- Aim for **high coverage on critical paths**, not 100% everywhere.

### Mocking
- Use `unittest.mock` or `pytest-mock`.
- Mock at **boundaries**, not internal implementation.
- Prefer **dependency injection** over patching.

---

## Documentation

### Docstrings
- Use **Google style** docstrings consistently.
- Document all public classes, methods, and functions.
- Include **Args**, **Returns**, **Raises**, and **Examples** sections.
- Document complex algorithms with inline comments.

### Type Documentation
- Let type hints serve as documentation where possible.
- Add docstrings for non-obvious type constraints.

---

## Imports & Dependencies

### Import Order (Enforced by Ruff)
1. Standard library
2. Third-party packages
3. Local application imports

### Import Style
- Use **absolute imports** for cross-package imports.
- Use **relative imports** only for intra-package imports: `from .service import UserService`.
- Import specific names: `from module import ClassName` (not `import module`).
- Never use wildcard imports (`from x import *`).
- Use `TYPE_CHECKING` block for import-only type hints to avoid circular imports:
  ```python
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from .models import User
  ```

---

## Security

### Secrets Management
- **Never** hardcode credentials, API keys, or secrets.
- Use environment variables via `pydantic-settings` or similar.
- Use secret management services in production.

### Input Validation
- Validate and sanitize **all** external inputs.
- Implement rate limiting on public APIs.

---

## SQLAlchemy 2.0 (Modern ORM)

### DeclarativeBase & Type-Annotated Models
- Use **`DeclarativeBase`** instead of legacy `declarative_base()`.
- Use **`Mapped[T]`** and **`mapped_column()`** for all column definitions.
- Never use legacy `Column()` syntax.
- Example:
  ```python
  from sqlalchemy import String, ForeignKey
  from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

  class Base(DeclarativeBase):
      pass

  class User(Base):
      __tablename__ = "users"

      id: Mapped[int] = mapped_column(primary_key=True)
      name: Mapped[str] = mapped_column(String(100))
      email: Mapped[str | None] = mapped_column(String(255), unique=True)
      posts: Mapped[list["Post"]] = relationship(back_populates="author")

  class Post(Base):
      __tablename__ = "posts"

      id: Mapped[int] = mapped_column(primary_key=True)
      title: Mapped[str] = mapped_column(String(200))
      author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
      author: Mapped["User"] = relationship(back_populates="posts")
  ```

### Session & Query Patterns
- Use **`select()`** statements instead of legacy `Query` API.
- Use **`Session.execute()`** with `select()` for queries.
- Use **`Session.scalars()`** for single-column results.
- Use **async sessions** (`AsyncSession`) for async code.
- Example:
  ```python
  from sqlalchemy import select
  from sqlalchemy.orm import Session

  def get_user_by_email(session: Session, email: str) -> User | None:
      stmt = select(User).where(User.email == email)
      return session.scalar(stmt)

  def get_active_users(session: Session) -> list[User]:
      stmt = select(User).where(User.is_active == True).order_by(User.name)
      return list(session.scalars(stmt))
  ```

### Relationship Patterns
- Always use **`Mapped[]`** type hints for relationships.
- Use **`back_populates`** instead of `backref` for explicit bidirectional relationships.
- Define **`relationship()`** with proper typing for IDE support.

### Deprecated SQLAlchemy Patterns to Avoid
- ❌ `declarative_base()` → ✅ `DeclarativeBase`
- ❌ `Column(Integer, ...)` → ✅ `Mapped[int] = mapped_column(...)`
- ❌ `session.query(User)` → ✅ `session.execute(select(User))`
- ❌ `backref="posts"` → ✅ `back_populates="posts"`
- ❌ `relationship("Post")` without type → ✅ `Mapped[list["Post"]] = relationship(...)`

---

## Observability, Monitoring & Tracing

### Structured Logging (Critical)
- Use **`structlog`** for structured, contextual logging throughout the application.
- Log in **JSON format** for production (machine-parseable).
- Include **correlation IDs** (trace_id, request_id) in all log entries.
- Use **log levels** appropriately: DEBUG for development, INFO for normal operations, WARNING for recoverable issues, ERROR for failures.
- Never log sensitive data (passwords, tokens, PII).
- **Integrate `structlog` with `contextvars`** for automatic context injection.
- Example:
  ```python
  import structlog
  from contextvars import ContextVar

  request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

  def add_context(logger, method_name, event_dict):
      if request_id := request_id_ctx.get():
          event_dict["request_id"] = request_id
      return event_dict

  structlog.configure(
      processors=[
          add_context,
          structlog.processors.JSONRenderer(),
      ]
  )

  logger = structlog.get_logger()

  async def process_request(request_id: str, user_id: str) -> None:
      request_id_ctx.set(request_id)
      log = logger.bind(user_id=user_id)
      log.info("processing_request_started")
      try:
          result = await do_work()
          log.info("processing_request_completed", result_count=len(result))
      except Exception as e:
          log.error("processing_request_failed", error=str(e), exc_info=True)
          raise
  ```

### Distributed Tracing
- Use **OpenTelemetry** for distributed tracing across services.
- Instrument all HTTP clients, database calls, and external service calls.
- Propagate **trace context** across service boundaries.
- Create **meaningful spans** for business operations.
- Add **span attributes** with relevant context (user_id, entity_id, operation_type).
- Example:
  ```python
  from opentelemetry import trace

  tracer = trace.get_tracer(__name__)

  async def fetch_user_data(user_id: str) -> UserData:
      with tracer.start_as_current_span("fetch_user_data") as span:
          span.set_attribute("user.id", user_id)
          data = await database.get_user(user_id)
          span.set_attribute("user.found", data is not None)
          return data
  ```

### Metrics & Instrumentation
- Export metrics via **OpenTelemetry** or **Prometheus** client.
- Track **RED metrics**: Rate, Errors, Duration for all endpoints.
- Instrument **custom business metrics** (orders processed, cache hit rates).
- Use **histograms** for latency measurements, **counters** for events.
- Add **labels/dimensions** for filtering (endpoint, status_code, service).

### Health Checks & Readiness
- Implement **/health** and **/ready** endpoints for all services.
- Health checks should verify **dependency connectivity** (database, cache, external APIs).
- Use **liveness probes** for crash detection, **readiness probes** for traffic routing.
- Return structured health status with component details.

### Error Tracking & Alerting
- Integrate with error tracking services (Sentry, Rollbar, etc.).
- Capture **exception context**: stack traces, request data, user context.
- Set up **alerting thresholds** for error rates and latency percentiles.
- Use **structured error codes** for categorization and filtering.

### Observability Best Practices
- **Instrument first, optimize later**: Add observability before performance tuning.
- Use **context propagation** to correlate logs, traces, and metrics.
- Implement **request-scoped context** for automatic correlation ID injection.
- Design **dashboards** around user journeys and SLOs.
- Store **audit logs** separately for compliance and debugging.
- Use **sampling** for high-volume traces to control costs.

---

## Performance

### Optimization Guidelines
- **Profile before optimizing** using `cProfile`, `py-spy`, or OpenTelemetry traces.
- Use `__slots__` for memory-critical classes.
- Use **generators** and **async generators** for large data streams to minimize memory usage.
- Cache expensive computations with `@functools.lru_cache` or `@functools.cache`.
- Use **`functools.cached_property`** for expensive computed properties.
- Use **async/await** for I/O-bound operations.
- Leverage **database query optimization**: proper indexing, query planning, connection pooling.
- **Prefer comprehensions** over `map()`/`filter()` for performance and readability (PEP 709).
- Use **`orjson`** instead of stdlib `json` for high-performance JSON serialization.

### Memory Management (Long-Running Services)
- Use **`tracemalloc`** or OpenTelemetry memory metrics for debugging leaks.
- Avoid accumulating closures in async loops.
- Prefer **async generators with explicit cleanup** (`aclosing()`).
- Use `__slots__` for frequently instantiated objects.
- Monitor memory usage in production dashboards.

### Standard Library Preferences
- Use **`tomllib`** (stdlib) for TOML parsing.
- Use **`zoneinfo.ZoneInfo`** for timezone handling:
  ```python
  from datetime import datetime
  from zoneinfo import ZoneInfo

  now_utc = datetime.now(ZoneInfo("UTC"))
  now_pacific = now_utc.astimezone(ZoneInfo("America/Los_Angeles"))
  ```
- Avoid `pickle` for anything crossing trust boundaries.

---

## Git & Version Control

### Commit Practices
- Write **conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Keep commits atomic and focused.
- Reference issue numbers: `feat: add user auth (#123)`.

### Branch Strategy
- Use feature branches: `feature/<name>`, `fix/<name>`.
- Keep PRs small and reviewable (< 400 lines ideally).
- Require code review before merging.
