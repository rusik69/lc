package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2220,
			Title:       "Python Testing and DevOps",
			Description: "Master pytest, unittest, mocking, property-based testing, CI/CD, Docker, packaging, and deployment patterns.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Testing Frameworks CI CD Docker and Deployment",
					Content: `Comprehensive testing and DevOps practices are essential for production Python applications.

**pytest — Modern Testing Framework:**

Test Discovery:
  Files: test_*.py or *_test.py
  Functions: test_*
  Classes: Test* (no __init__)
  
Basic Assertions:
  assert x == y
  assert x != y
  assert x is None
  assert x in collection
  assert isinstance(x, Type)
  
  pytest.raises(ExceptionType) — expect exception
  pytest.warns(WarningType) — expect warning
  pytest.approx(value) — floating point comparison

Fixtures:
  @pytest.fixture — reusable setup/teardown
  
  Scopes:
    function (default): Each test
    class: Once per test class
    module: Once per module
    package: Once per package
    session: Once per test run
    
  Yield fixtures:
    yield value — provides value to test
    Code after yield = teardown
    
  Fixture dependencies:
    Fixtures can request other fixtures
    Automatic dependency injection
    
  conftest.py:
    Shared fixtures across tests in directory
    Automatically loaded by pytest
    Can be nested in subdirectories

  @pytest.fixture(autouse=True) — automatic use
  
  Built-in fixtures:
    tmp_path: Temporary directory
    capsys: Capture stdout/stderr
    monkeypatch: Dynamic patching
    request: Test request info

Parameterization:
  @pytest.mark.parametrize("arg1,arg2", [(1, 2), (3, 4)])
  Multiple parameterize decorators: Cartesian product
  indirect=True: Parameterize via fixture

Markers:
  @pytest.mark.skip(reason="...")
  @pytest.mark.skipif(condition, reason="...")
  @pytest.mark.xfail(reason="...") — expected failure
  @pytest.mark.slow — custom marker
  
  Register markers in pytest.ini or pyproject.toml
  Run specific markers: pytest -m "slow"

Plugins:
  pytest-cov: Code coverage
  pytest-xdist: Parallel execution
  pytest-asyncio: Async test support
  pytest-mock: Mocker fixture
  pytest-benchmark: Performance testing
  pytest-randomly: Randomize test order
  pytest-timeout: Test timeouts
  pytest-httpserver: HTTP mock server
  pytest-freezegun: Time mocking

Configuration (pyproject.toml):
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  addopts = "-v --tb=short"
  markers = ["slow: marks tests as slow"]
  filterwarnings = ["error"]

**unittest — Standard Library Testing:**

TestCase:
  class TestX(unittest.TestCase)
  setUp() / tearDown() — per test
  setUpClass() / tearDownClass() — per class
  
Assertions:
  self.assertEqual(a, b)
  self.assertNotEqual(a, b)
  self.assertTrue(x) / self.assertFalse(x)
  self.assertIs(a, b) / self.assertIsNone(x)
  self.assertIn(a, b) / self.assertNotIn(a, b)
  self.assertRaises(Error, func, *args)
  self.assertAlmostEqual(a, b, places=7)
  self.assertGreater(a, b) / self.assertLess(a, b)
  self.assertRegex(text, pattern)
  self.assertCountEqual(a, b) — same elements, any order

**Mocking:**

unittest.mock:
  Mock() — general mock object
  MagicMock() — mock with magic methods
  patch() — replace objects temporarily
  
  Mock attributes:
    mock.return_value — value to return
    mock.side_effect — function or exception
    mock.call_count — number of calls
    mock.call_args — last call arguments
    mock.call_args_list — all call arguments
    
  Assertions:
    mock.assert_called()
    mock.assert_called_once()
    mock.assert_called_with(*args, **kwargs)
    mock.assert_called_once_with(*args, **kwargs)
    mock.assert_not_called()
    mock.assert_any_call(*args, **kwargs)
    
  patch:
    @patch('module.ClassName')
    @patch.object(ClassName, 'method')
    @patch.dict(dict_obj, {'key': 'value'})
    
    with patch('module.func') as mock_func:
        mock_func.return_value = 42
        
  spec:
    Mock(spec=RealClass) — restrict to real interface
    create_autospec(RealClass) — recursive spec

**Property-Based Testing (Hypothesis):**

  @given(st.integers(), st.text())
  @settings(max_examples=100)
  
  Strategies:
    st.integers(min_value=0, max_value=100)
    st.floats(allow_nan=False)
    st.text(min_size=1, max_size=100)
    st.lists(st.integers(), min_size=1)
    st.dictionaries(st.text(), st.integers())
    st.one_of(st.integers(), st.text())
    st.builds(MyClass, name=st.text())
    
  Features:
    Shrinking: Find minimal failing example
    Database: Remember failing examples
    Profiles: Configure for CI vs local
    Stateful testing: Test state machines

**Code Coverage:**

  pytest --cov=package --cov-report=html
  
  Coverage types:
    Line coverage: Which lines executed
    Branch coverage: Which branches taken
    Condition coverage: Boolean expression outcomes
    
  Configuration (.coveragerc):
    [run]
    source = src
    omit = tests/*, */__init__.py
    branch = True
    
    [report]
    fail_under = 80
    exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:"]

**Docker for Python:**

Dockerfile best practices:
  Multi-stage builds
  Non-root user
  Pin dependency versions
  .dockerignore file
  Layer caching optimization
  Virtual environment in container
  
  Base images:
    python:3.12-slim — small Debian
    python:3.12-alpine — smallest (musl libc)
    python:3.12-bookworm — full Debian

  Dependency installation:
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    
  Or with uv:
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
    COPY pyproject.toml uv.lock ./
    RUN uv sync --frozen
    COPY . .

Docker Compose:
  Multi-container applications
  Service dependencies
  Volume mounts for development
  Environment variables
  Health checks

**CI/CD:**

GitHub Actions:
  .github/workflows/test.yml
  Matrix testing (multiple Python versions)
  Caching pip/uv dependencies
  Publishing to PyPI
  
  Steps:
    1. Checkout code
    2. Set up Python
    3. Install dependencies
    4. Run linters (ruff, mypy)
    5. Run tests with coverage
    6. Upload coverage report
    7. Build and publish (on release)

**Python Packaging:**

pyproject.toml (modern standard):
  [build-system]
  [project] — metadata
  [project.scripts] — CLI entry points
  [project.optional-dependencies] — extras
  
  Build backends:
    setuptools: Traditional
    hatchling: Modern (Hatch)
    flit_core: Minimal
    pdm-backend: PDM
    maturin: Rust extensions

Package Structure:
  src layout:
    src/mypackage/__init__.py
    src/mypackage/module.py
    tests/test_module.py
    pyproject.toml
    
  Flat layout:
    mypackage/__init__.py
    mypackage/module.py
    tests/test_module.py
    pyproject.toml

Publishing:
  Build: python -m build
  Upload: twine upload dist/*
  Test PyPI: twine upload --repository testpypi dist/*

**Type Checking (mypy):**

Configuration:
  [tool.mypy]
  python_version = "3.12"
  strict = true
  warn_return_any = true
  
Plugin support:
  mypy-django
  pydantic mypy plugin
  sqlalchemy mypy plugin

Common patterns:
  reveal_type(x) — debug type inference
  cast(Type, value) — type cast
  TYPE_CHECKING — import-only types
  @overload — overloaded functions

**Linting (Ruff):**

  Extremely fast Python linter (Rust-based)
  Replaces: flake8, isort, pyupgrade, etc.
  
  [tool.ruff]
  line-length = 88
  target-version = "py312"
  
  [tool.ruff.lint]
  select = ["E", "F", "I", "N", "W", "UP"]
  
  ruff check . — lint
  ruff format . — format (like Black)

**Logging Best Practices:**

  import logging
  logger = logging.getLogger(__name__)
  
  Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  
  Structured logging:
    structlog — structured log events
    JSON output for log aggregation
    
  Configuration:
    logging.config.dictConfig({...})
    Handlers: StreamHandler, FileHandler, RotatingFileHandler
    Formatters: Custom format strings
    Filters: Contextual filtering`,
					CodeExamples: `# Python testing and DevOps examples

import os
import sys
import json
import time
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from unittest.mock import MagicMock

# ============================================================
# Test Framework Implementation
# ============================================================

class TestResult:
    """Stores test execution results."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []
        self.skipped = []
        self.duration = 0.0
    
    @property
    def total(self):
        return len(self.passed) + len(self.failed) + len(self.errors) + len(self.skipped)
    
    @property
    def success_rate(self):
        if self.total == 0:
            return 0.0
        return len(self.passed) / self.total * 100
    
    def summary(self) -> str:
        return (
            f"{self.total} tests: "
            f"{len(self.passed)} passed, "
            f"{len(self.failed)} failed, "
            f"{len(self.errors)} errors, "
            f"{len(self.skipped)} skipped "
            f"({self.duration:.3f}s)"
        )


class TestCase:
    """Base test case class."""
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def assertEqual(self, a, b, msg=None):
        if a != b:
            raise AssertionError(msg or f"{a!r} != {b!r}")
    
    def assertNotEqual(self, a, b, msg=None):
        if a == b:
            raise AssertionError(msg or f"{a!r} == {b!r}")
    
    def assertTrue(self, x, msg=None):
        if not x:
            raise AssertionError(msg or f"{x!r} is not truthy")
    
    def assertFalse(self, x, msg=None):
        if x:
            raise AssertionError(msg or f"{x!r} is not falsy")
    
    def assertIsNone(self, x, msg=None):
        if x is not None:
            raise AssertionError(msg or f"{x!r} is not None")
    
    def assertIsNotNone(self, x, msg=None):
        if x is None:
            raise AssertionError(msg or "unexpectedly None")
    
    def assertIn(self, member, container, msg=None):
        if member not in container:
            raise AssertionError(msg or f"{member!r} not in {container!r}")
    
    def assertNotIn(self, member, container, msg=None):
        if member in container:
            raise AssertionError(msg or f"{member!r} in {container!r}")
    
    def assertIsInstance(self, obj, cls, msg=None):
        if not isinstance(obj, cls):
            raise AssertionError(
                msg or f"{obj!r} is not an instance of {cls!r}")
    
    def assertRaises(self, exc_type):
        return _AssertRaisesContext(exc_type)
    
    def assertAlmostEqual(self, a, b, places=7, msg=None):
        if round(abs(a - b), places) != 0:
            raise AssertionError(
                msg or f"{a} != {b} within {places} places")
    
    def assertGreater(self, a, b, msg=None):
        if not a > b:
            raise AssertionError(msg or f"{a!r} not greater than {b!r}")
    
    def assertLess(self, a, b, msg=None):
        if not a < b:
            raise AssertionError(msg or f"{a!r} not less than {b!r}")


class _AssertRaisesContext:
    def __init__(self, exc_type):
        self._exc_type = exc_type
        self.exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"{self._exc_type.__name__} not raised")
        if not issubclass(exc_type, self._exc_type):
            return False
        self.exception = exc_val
        return True


class TestRunner:
    """Discovers and runs tests."""
    
    def __init__(self, verbosity=1):
        self.verbosity = verbosity
    
    def discover(self, test_class: type) -> List[str]:
        return [
            name for name in dir(test_class)
            if name.startswith('test_') and callable(getattr(test_class, name))
        ]
    
    def run(self, test_classes: List[type]) -> TestResult:
        result = TestResult()
        start = time.time()
        
        for cls in test_classes:
            test_names = self.discover(cls)
            
            try:
                cls.setUpClass()
            except Exception as e:
                result.errors.append((f"{cls.__name__}.setUpClass", str(e)))
                continue
            
            for name in test_names:
                instance = cls()
                full_name = f"{cls.__name__}.{name}"
                
                # Check for skip marker
                method = getattr(instance, name)
                if hasattr(method, '_skip'):
                    result.skipped.append((full_name, method._skip_reason))
                    if self.verbosity > 0:
                        print(f"  SKIP {full_name}: {method._skip_reason}")
                    continue
                
                try:
                    instance.setUp()
                    method()
                    instance.tearDown()
                    result.passed.append(full_name)
                    if self.verbosity > 0:
                        print(f"  PASS {full_name}")
                except AssertionError as e:
                    result.failed.append((full_name, str(e)))
                    if self.verbosity > 0:
                        print(f"  FAIL {full_name}: {e}")
                except Exception as e:
                    result.errors.append((full_name, str(e)))
                    if self.verbosity > 0:
                        print(f"  ERROR {full_name}: {e}")
            
            try:
                cls.tearDownClass()
            except Exception:
                pass
        
        result.duration = time.time() - start
        
        if self.verbosity > 0:
            print(f"\n{result.summary()}")
        
        return result


def skip(reason=""):
    """Mark test to be skipped."""
    def decorator(func):
        func._skip = True
        func._skip_reason = reason
        return func
    return decorator


def parametrize(argnames, argvalues):
    """Parametrize a test function."""
    def decorator(func):
        func._parametrize = (argnames, argvalues)
        return func
    return decorator


# ============================================================
# Fixture System
# ============================================================

class FixtureManager:
    """Manages test fixtures with dependency injection."""
    
    def __init__(self):
        self._fixtures: Dict[str, dict] = {}
        self._cache: Dict[str, Any] = {}
    
    def register(self, name: str, func: Callable, scope: str = 'function'):
        self._fixtures[name] = {
            'func': func,
            'scope': scope,
            'dependencies': self._get_dependencies(func),
        }
    
    def _get_dependencies(self, func: Callable) -> List[str]:
        import inspect
        sig = inspect.signature(func)
        return [
            name for name in sig.parameters
            if name != 'self' and name in self._fixtures
        ]
    
    def resolve(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        
        fixture = self._fixtures.get(name)
        if not fixture:
            raise ValueError(f"Unknown fixture: {name}")
        
        # Resolve dependencies
        deps = {}
        for dep_name in fixture['dependencies']:
            deps[dep_name] = self.resolve(dep_name)
        
        # Execute fixture
        result = fixture['func'](**deps)
        
        # Cache based on scope
        if fixture['scope'] in ('module', 'session'):
            self._cache[name] = result
        
        return result
    
    def clear_cache(self, scope: str = None):
        if scope is None:
            self._cache.clear()
        else:
            to_remove = [
                name for name, fix in self._fixtures.items()
                if fix['scope'] == scope and name in self._cache
            ]
            for name in to_remove:
                del self._cache[name]


# ============================================================
# Mock Objects
# ============================================================

class Mock:
    """Simple mock object implementation."""
    
    def __init__(self, spec=None, return_value=None, side_effect=None):
        self._spec = spec
        self._return_value = return_value
        self._side_effect = side_effect
        self._calls = []
        self._children = {}
    
    def __call__(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        
        if self._side_effect:
            if isinstance(self._side_effect, type) and issubclass(self._side_effect, Exception):
                raise self._side_effect()
            if callable(self._side_effect):
                return self._side_effect(*args, **kwargs)
            raise self._side_effect
        
        return self._return_value
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        
        if self._spec:
            if not hasattr(self._spec, name):
                raise AttributeError(
                    f"Mock spec {self._spec} has no attribute '{name}'")
        
        if name not in self._children:
            self._children[name] = Mock()
        return self._children[name]
    
    @property
    def call_count(self):
        return len(self._calls)
    
    @property
    def call_args(self):
        if self._calls:
            return self._calls[-1]
        return None
    
    @property
    def call_args_list(self):
        return self._calls[:]
    
    def assert_called(self):
        if not self._calls:
            raise AssertionError("Expected to be called")
    
    def assert_called_once(self):
        if len(self._calls) != 1:
            raise AssertionError(
                f"Expected to be called once, called {len(self._calls)} times")
    
    def assert_called_with(self, *args, **kwargs):
        if not self._calls:
            raise AssertionError("Not called")
        actual_args, actual_kwargs = self._calls[-1]
        if actual_args != args or actual_kwargs != kwargs:
            raise AssertionError(
                f"Expected call({args}, {kwargs}), "
                f"got call({actual_args}, {actual_kwargs})")
    
    def assert_not_called(self):
        if self._calls:
            raise AssertionError(
                f"Expected not to be called, called {len(self._calls)} times")
    
    def reset_mock(self):
        self._calls.clear()
        for child in self._children.values():
            child.reset_mock()


@contextmanager
def patch(target: str, mock_obj=None):
    """Context manager to temporarily replace an attribute."""
    parts = target.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid target: {target}")
    
    module_path, attr_name = parts
    
    # Import the module
    module = sys.modules.get(module_path)
    if module is None:
        __import__(module_path)
        module = sys.modules[module_path]
    
    original = getattr(module, attr_name)
    replacement = mock_obj or Mock()
    
    setattr(module, attr_name, replacement)
    try:
        yield replacement
    finally:
        setattr(module, attr_name, original)


# ============================================================
# Coverage Tracker
# ============================================================

class CoverageTracker:
    """Simple code coverage tracker."""
    
    def __init__(self):
        self._executed_lines: Dict[str, set] = {}
        self._total_lines: Dict[str, set] = {}
    
    def record_execution(self, filename: str, line_number: int):
        if filename not in self._executed_lines:
            self._executed_lines[filename] = set()
        self._executed_lines[filename].add(line_number)
    
    def set_total_lines(self, filename: str, lines: set):
        self._total_lines[filename] = lines
    
    def get_coverage(self, filename: str) -> float:
        executed = len(self._executed_lines.get(filename, set()))
        total = len(self._total_lines.get(filename, set()))
        if total == 0:
            return 100.0
        return (executed / total) * 100
    
    def get_overall_coverage(self) -> float:
        total_executed = sum(len(lines) for lines in self._executed_lines.values())
        total_lines = sum(len(lines) for lines in self._total_lines.values())
        if total_lines == 0:
            return 100.0
        return (total_executed / total_lines) * 100
    
    def get_uncovered_lines(self, filename: str) -> set:
        total = self._total_lines.get(filename, set())
        executed = self._executed_lines.get(filename, set())
        return total - executed
    
    def report(self) -> str:
        lines = ["Name                    Stmts   Miss  Cover"]
        lines.append("-" * 50)
        
        for filename in sorted(self._total_lines.keys()):
            total = len(self._total_lines[filename])
            executed = len(self._executed_lines.get(filename, set()))
            miss = total - executed
            cover = self.get_coverage(filename)
            name = filename.split('/')[-1]
            lines.append(f"{name:<24}{total:>5}{miss:>7}{cover:>6.0f}%")
        
        lines.append("-" * 50)
        overall = self.get_overall_coverage()
        lines.append(f"{'TOTAL':<24}{'':>5}{'':>7}{overall:>6.0f}%")
        
        return '\n'.join(lines)


# ============================================================
# Docker Compose Configuration Builder
# ============================================================

class DockerCompose:
    """Build Docker Compose configurations."""
    
    def __init__(self, version: str = "3.8"):
        self._version = version
        self._services: Dict[str, dict] = {}
        self._volumes: Dict[str, dict] = {}
        self._networks: Dict[str, dict] = {}
    
    def add_service(self, name: str, image: str = None, build: str = None,
                    ports: list = None, environment: dict = None,
                    volumes: list = None, depends_on: list = None,
                    command: str = None, healthcheck: dict = None):
        service = {}
        
        if image:
            service['image'] = image
        if build:
            service['build'] = build
        if ports:
            service['ports'] = ports
        if environment:
            service['environment'] = environment
        if volumes:
            service['volumes'] = volumes
        if depends_on:
            service['depends_on'] = depends_on
        if command:
            service['command'] = command
        if healthcheck:
            service['healthcheck'] = healthcheck
        
        self._services[name] = service
    
    def add_volume(self, name: str, driver: str = 'local'):
        self._volumes[name] = {'driver': driver}
    
    def add_network(self, name: str, driver: str = 'bridge'):
        self._networks[name] = {'driver': driver}
    
    def to_dict(self) -> dict:
        config = {'version': self._version}
        
        if self._services:
            config['services'] = self._services
        if self._volumes:
            config['volumes'] = self._volumes
        if self._networks:
            config['networks'] = self._networks
        
        return config
    
    def to_yaml(self) -> str:
        """Simple YAML-like output."""
        lines = [f"version: '{self._version}'", "", "services:"]
        
        for name, service in self._services.items():
            lines.append(f"  {name}:")
            for key, value in service.items():
                if isinstance(value, list):
                    lines.append(f"    {key}:")
                    for item in value:
                        lines.append(f"      - {item}")
                elif isinstance(value, dict):
                    lines.append(f"    {key}:")
                    for k, v in value.items():
                        lines.append(f"      {k}: {v}")
                else:
                    lines.append(f"    {key}: {value}")
        
        if self._volumes:
            lines.append("")
            lines.append("volumes:")
            for name in self._volumes:
                lines.append(f"  {name}:")
        
        return '\n'.join(lines)


# ============================================================
# Logging Configuration
# ============================================================

class StructuredLogger:
    """JSON structured logger."""
    
    def __init__(self, name: str, level: str = 'INFO'):
        self.name = name
        self.level = getattr(logging, level, logging.INFO)
        self._context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Add context fields."""
        new_logger = StructuredLogger(self.name)
        new_logger._context = {**self._context, **kwargs}
        new_logger.level = self.level
        return new_logger
    
    def _log(self, level: str, message: str, **kwargs):
        log_level = getattr(logging, level, 0)
        if log_level < self.level:
            return
        
        entry = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'level': level,
            'logger': self.name,
            'message': message,
            **self._context,
            **kwargs
        }
        
        print(json.dumps(entry))
    
    def debug(self, message: str, **kwargs):
        self._log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log('CRITICAL', message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        kwargs['traceback'] = traceback.format_exc()
        self._log('ERROR', message, **kwargs)


# ============================================================
# CI/CD Pipeline Configuration
# ============================================================

class CIPipeline:
    """CI/CD pipeline configuration builder."""
    
    def __init__(self, name: str):
        self.name = name
        self._stages: List[dict] = []
        self._env: Dict[str, str] = {}
        self._matrix: Dict[str, list] = {}
    
    def add_env(self, key: str, value: str):
        self._env[key] = value
    
    def set_matrix(self, **kwargs):
        self._matrix = kwargs
    
    def add_stage(self, name: str, steps: List[dict],
                  needs: list = None, condition: str = None):
        stage = {
            'name': name,
            'steps': steps,
        }
        if needs:
            stage['needs'] = needs
        if condition:
            stage['if'] = condition
        self._stages.append(stage)
    
    def to_github_actions(self) -> dict:
        workflow = {
            'name': self.name,
            'on': {'push': {'branches': ['main']}, 'pull_request': {}},
        }
        
        jobs = {}
        for stage in self._stages:
            job = {
                'runs-on': 'ubuntu-latest',
                'steps': [
                    {'uses': 'actions/checkout@v4'},
                ],
            }
            
            if self._matrix:
                job['strategy'] = {'matrix': self._matrix}
            
            for step in stage['steps']:
                job['steps'].append(step)
            
            if stage.get('needs'):
                job['needs'] = stage['needs']
            
            if stage.get('if'):
                job['if'] = stage['if']
            
            jobs[stage['name'].replace(' ', '-').lower()] = job
        
        workflow['jobs'] = jobs
        return workflow


class PackageConfig:
    """Python package configuration builder."""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.description = ""
        self.authors = []
        self.dependencies = []
        self.dev_dependencies = []
        self.python_requires = ">=3.10"
        self.entry_points = {}
    
    def add_dependency(self, pkg: str, version: str = ""):
        if version:
            self.dependencies.append(f"{pkg}>={version}")
        else:
            self.dependencies.append(pkg)
    
    def add_dev_dependency(self, pkg: str):
        self.dev_dependencies.append(pkg)
    
    def add_script(self, name: str, entry: str):
        self.entry_points[name] = entry
    
    def to_pyproject(self) -> str:
        lines = [
            "[build-system]",
            'requires = ["hatchling"]',
            'build-backend = "hatchling.build"',
            "",
            "[project]",
            f'name = "{self.name}"',
            f'version = "{self.version}"',
            f'description = "{self.description}"',
            f'requires-python = "{self.python_requires}"',
        ]
        
        if self.dependencies:
            lines.append("dependencies = [")
            for dep in self.dependencies:
                lines.append(f'    "{dep}",')
            lines.append("]")
        
        if self.dev_dependencies:
            lines.append("")
            lines.append("[project.optional-dependencies]")
            lines.append("dev = [")
            for dep in self.dev_dependencies:
                lines.append(f'    "{dep}",')
            lines.append("]")
        
        if self.entry_points:
            lines.append("")
            lines.append("[project.scripts]")
            for name, entry in self.entry_points.items():
                lines.append(f'{name} = "{entry}"')
        
        return '\n'.join(lines)`,
				},
			},
		},
	})
}
