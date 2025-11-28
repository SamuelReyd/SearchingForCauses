#!/bin/bash
source env/bin/activate
pip install -q -r requirements.txt
python -m ipykernel install --user --name=myenv
