SOFTWARE USED

1. Python 3.x
   - Download: https://www.python.org/downloads/
   - Install required packages:
     pip install pyspark pandas numpy streamlit streamlit-autorefresh matplotlib seaborn

2. PySpark (installed via pip, runs inside notebook)
   - No separate installation needed
   - Included in pip install above

3. Google Colab (for running index.ipynb)
   - No installation needed
   - Go to https://colab.research.google.com
   - Upload index.ipynb and run all cells in order

4. Streamlit (for running dashboard)
   - After running notebook and downloading CSVs:
     cd "Dashboard files"
     streamlit run app.py