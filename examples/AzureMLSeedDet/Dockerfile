FROM mcr.microsoft.com/azureml/curated/azureml-automl-dnn-vision-gpu:131

WORKDIR /app

# ENV QUART_APP=app.py
# ENV QUART_ENV=development
# ENV PYTHONUNBUFFERED True
# ENV PYTHONPATH=/app

COPY ./requirements.txt .

# RUN pip3 install torch==1.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# RUN pip3 install --no-cache-dir -r requirements.txt

COPY . ./

# CMD hypercorn -b :$PORT app:app
CMD ls && /bin/bash