FROM python:3.11
WORKDIR /code
COPY . /code
RUN pip install -r /code/requirements.txt
CMD ["streamlit", "run","--server.port","8000", "main.py"]