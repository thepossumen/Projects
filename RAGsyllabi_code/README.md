# Code repository for group work in DIT247

## Instructions for running project code:

### Google Colab
- Start a Google Colab session, free-tier with T4 GPU should suffice. 
- Open the file colab-code/retrieve_generate_entrypoint.ipynb
- Uncomment and run the first pip-install cell and then restart the session. 
- Re-comment the pip-installs
- Upload the following files to the session:
  - `colab_code/func_script.py`
  - `eval_dataset/update_eval_set.py` 
- Select the model (recommended: 1) under "Choose LLM-model to load"
- Run all
- Play around with queries under "Query construction" and parameters under "Main"