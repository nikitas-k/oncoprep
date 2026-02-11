# Contributing

Contributions are welcome! Here's how to get started.

## Development setup

```bash
git clone https://github.com/nikitas-k/oncoprep.git
cd oncoprep
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Code style

OncoPrep uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/oncoprep
```

## Testing

```bash
pytest tests/
```

## Workflow conventions

Every new workflow **must** follow the Nipype factory pattern:

```python
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def init_my_wf(*, keyword_only_args) -> Workflow:
    """One-liner docstring.

    Parameters
    ----------
    arg_name : type
        Description.

    Returns
    -------
    Workflow
        The constructed workflow.
    """
    workflow = Workflow(name="my_wf")

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in1"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out1"]),
        name="outputnode",
    )

    # ... processing nodes and connections ...

    return workflow
```

## Type hints

Use `typing` module syntax for Python 3.9 compatibility:

```python
# Yes
from typing import Optional, List
def func(x: Optional[List[str]]) -> str: ...

# No (3.10+ only)
def func(x: list[str] | None) -> str: ...
```

## Documentation

Build docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html
```

Preview at `docs/_build/html/index.html`.
