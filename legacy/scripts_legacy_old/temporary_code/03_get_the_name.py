# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/03_get_the_name.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import time 
time.ctime(time.time())

# %%
import json
import os
import requests
from notebook import notebookapp

def get_notebook_name():
    # Get the connection file and extract the kernel ID
    connection_file = os.path.basename(get_ipython().config['IPKernelApp']['connection_file'])
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    # Find the notebook server and request the API for sessions
    for srv in notebookapp.list_running_servers():
        try:
            response = requests.get(f"{srv['url']}api/sessions", params={'token': srv.get('token', '')})
            for sess in json.loads(response.text):
                if sess['kernel']['id'] == kernel_id:
                    return sess['notebook']['path'].split('/')[-1]
        except Exception as e:
            print(f"An error occurred: {e}")
    
    return None

notebook_name = get_notebook_name()
print(f"Notebook name: {notebook_name}")

# %%
!pip install requests

