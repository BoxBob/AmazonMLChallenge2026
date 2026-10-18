import json, re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
cells = []
for block in re.split(r"^(?=# %%)", text, flags=re.M):
    if not block.strip():
        continue
    header, _, body = block.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in header:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"id": f"c{len(cells):02d}", "cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
    else:
        cells.append({"id": f"c{len(cells):02d}", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(cells), "cells ->", dst)
