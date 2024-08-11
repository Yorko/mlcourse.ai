<div align="center">
<img src='_static/img/ods_stickers.jpg'>
<a href=https://mlcourse.ai>mlcourse.ai</a> – Open Machine Learning course. Bonus assignments
</div>

# Welcome

Hi! In this [mlcourse.ai JupyterBook](_build/html/index.html) you'll find your Bonus Assignments. The book is organized in 10 topics closely following the publicly available [mlcourse.ai](https://mlcourse.ai/book/index.html) self-paced learning program. In each section, you'll find a short intro to the topic (with links to the corresponding course materials), an assignment, and a solution to the assignment. 

To launch the JupyterBook, go to the `_build/html/` folder and open [`index.html`](_build/html/index.html) with your favourite browser. Note that JupyterBook is read-only. So the best way to work with the assignments is to run `jupyter-notebook` (not to be confused with `jupyter-book`) from the root directory `mlcourse_ai_bonus_jupyter_book`, and then make a copy of the notebook to work with as shown below. 

<div align="center">
<img src='_static/img/bonus_assignments_jupyter.png'>
</div>
<br>

## Managing dependencies

You can either install the dependencies manually from the `pyproject.toml` file or use [Poetry](https://python-poetry.org/).

- `poetry install` – installs the dependencies from the provided `poetry.lock` file
- `poetry run jupyter-notebook` – runs `jupyter-notebook` in the virtual environment managed py Poetry   
- `poetry run jupyter-book build .` – builds the Jupyter book and wtites static HTML files to the `_build` folder. 

**Thanks for supporting the course and happy learning!**