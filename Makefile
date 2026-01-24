install:
	uv sync

build:
	uv run jb build mlcourse_ai_jupyter_book

check_links:
	uv run jupyter-book build mlcourse_ai_jupyter_book/ --builder linkcheck

deploy:
	uv run ghp-import -n -p -f mlcourse_ai_jupyter_book/_build/html -c mlcourse.ai