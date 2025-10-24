# Repository Guidelines

## Project Structure & Module Organization
- `course_planning/` captures syllabi, slot plans, and weekly teaching templates; keep new planning docs sorted by module and week.
- `weekly_plans/` drills down to assessments like `1-Formative-Test-I-Oct-10th/question_papers/`; mirror this hierarchy when adding new question sets or answer keys.
- `labs/` stores executable learning materials (`simple_mnist_ui.py`, `gradioUI.py`, `exercise-1/` assets) plus bundled environments (`srnenv/`, `exercise-1/mnist-env/`).
- Long-form references, slides, and media belong in `_attachments/`, `books/`, and `architecture_diagrams/`; link to large binaries rather than duplicating them elsewhere.

## Build, Test, and Development Commands
- Create a sandboxed interpreter with `python3 -m venv .venv && source .venv/bin/activate` (or reuse `labs/srnenv/bin/activate` when exploring MNIST labs).
- Install the common toolchain: `pip install tensorflow==2.20.0 numpy==2.2.6 opencv-python==4.12.0 matplotlib==3.10.5 gradio==4.44.0 pytest black ruff`.
- Run desktop demos via `python3 labs/simple_mnist_ui.py`; launch the Gradio variant with `python3 labs/gradioUI.py` once dependencies resolve.
- Before packaging assessments, render previews with `pandoc question_papers/Question-paper-4.md -o Question-paper-4.pdf` (adjust filenames as needed).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, `snake_case` modules, and descriptive filenames (e.g., `mnist_canvas.py`, `week03_activation_plan.md`).
- Add module docstrings summarizing teaching intent; prefer explicit type hints in reusable utilities under `labs/`.
- Format notebook exports and scripts using `black labs/` and lint with `ruff labs/`; run both prior to sharing with students.

## Testing Guidelines
- Use `pytest` for automated checks; co-locate new suites under `labs/tests/` or beside the code they validate.
- Name tests `test_<feature>.py` and include quick smoke cases that load models and perform a single inference.
- Execute `pytest -q` before distributing lab updates; capture GPU/CPU-specific notes in the module README.

## Commit & Pull Request Guidelines
- Repository snapshots ship without VCS metadata; adopt imperative Conventional Commit headers such as `labs: add dropout exercise answer key`.
- Reference related planning docs by relative path inside the description (`course_planning/weekly_plans/week9-module3-features/...`).
- Summaries should list verification steps (commands run, files exported) and attach screenshots for UI changes when possible.

## Security & Data Handling
- Never store student identifiers or raw assessment answers outside protected folders; redact before sharing via git.
- Keep datasets, large media, and generated models in `_attachments/` or versioned storage, committing only lightweight samples.
- Exclude virtual environments (`.venv/`, `labs/srnenv/`) and local caches through workspace `.gitignore` entries.
