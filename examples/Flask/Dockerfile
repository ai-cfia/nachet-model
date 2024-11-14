FROM python:3.11-bullseye

WORKDIR /app

# ENV QUART_APP=app.py
# ENV QUART_ENV=development
# ENV PYTHONUNBUFFERED True
# ENV PYTHONPATH=/app

COPY ./requirements.txt .

RUN pip3 install torch==2.2.0+cpu torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . ./

# CMD hypercorn -b :$PORT app:app
# EXPOSE 5000
# RUN FLASK_APP=app.py flask run
