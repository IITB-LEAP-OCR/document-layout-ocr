# document-ocr
Layout preserving OCR for documents. Includes text, tables and figures. Useful for LEAP OCR and Bhashini apps API call.


### Step 1 : Create Virtual Environment 
Make sure you are using Python 3.10 and create a virtual environment to install upcoming dependencies
```
python3 -m venv <myenvpath>
```


### Step 2 : Install Requirements
Use this virtual environment to install the following dependencies
```
pip install -r requirements.txt
```

### Step 3 : Download Models
From the release section download the two models. Place figure-detector model in 'figures/model' and place sprint.pt for table strcuture recogniiton in 'tables/model' directory 

### Step 4 : Run the pipeline
Use main.py to set the input file parameters, output set name, language, table, and figures flag and execute as follows.
```
python3 main.py
```

### Step 5 : Using the UI
You can also use the streamlit UI to execute the pipeline and download the compressed output. 
```
streamlit run app.py
```
