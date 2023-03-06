FROM python:3.8

# streamlit working dir
RUN mkdir /app
WORKDIR /app

# copy everything including the pip requirements file 
COPY . /app

# pip install the requirements 
RUN pip3 install -r requirements.txt

# use streamlit by default
ENTRYPOINT ["streamlit", "run"]

# command to run - this is run when docker image is opened (by streamlit)
# equiv to `streamlit run Overview.py` 
CMD ["Hospital_Efficiency_Project_Home.py"]



