# document-ocr
Layout preserving OCR for documents. Includes text, tables and figures. Useful for LEAP OCR and Bhashini apps API call.


### Step 1 : Install Requirements
You may create and use virtual environment to install the followh gdependencies
```
pip install -r requirements.in
```

### Step 2 : Download Models
From the release section download the two models. Place figure-detector model in 'figures/model' and place sprint.pt for table strcuture recogniiton in 'tables/model' directory 

### Step 3 : Run the pipeline
Use main.py to set the parameters of input file, output set name, language, table and figures flag and execute as follows.
```
python3 main.py
```

### Step 4 : Using the UI
You can also use the streamlit UI to execute the pipeline and download the compressed output. 
```
streamlit run app.py
```