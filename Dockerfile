FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./MLhousingPrices /code/MLhousingPrices

# expose port
EXPOSE 80

# run server
CMD ["uvicorn", "MLhousingPrices.app:app", "--host", "0.0.0.0", "--port", "80"]