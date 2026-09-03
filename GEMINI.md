# Project Context

Main content is located in `mlcourse_ai_jupyter_book/book/` as executable markdown files, e.g. `mlcourse_ai_jupyter_book/book/topic01` is the content for Topic 1.

## Build & Deploy

`mlcourse.ai` is built from `mlcourse_ai_jupyter_book` with:

```bash
uv run jb build mlcourse_ai_jupyter_book
```

Deployment is explained here: https://jupyterbook.org/publish/gh-pages.html

A separate branch `gh-pages` exists for the built artifacts. The following command is used to update the build:

```bash
uv run ghp-import -n -p -f mlcourse_ai_jupyter_book/_build/html -c mlcourse.ai
```
