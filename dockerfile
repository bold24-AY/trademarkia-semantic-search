FROM python:3.10

WORKDIR /app

# copy project files
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose api port
EXPOSE 8000

# start api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]