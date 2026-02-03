# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/09_name_of_jupyter.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
from IPython.display import Javascript, display
import json

def get_notebook_name():
    js_code = """
    IPython.notebook.kernel.execute('notebook_name = "' + IPython.notebook.notebook_name + '"')
    """
    display(Javascript(js_code))

get_notebook_name()

# %%
!pip install ipynbname

