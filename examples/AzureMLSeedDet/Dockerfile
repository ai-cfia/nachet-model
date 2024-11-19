FROM mcr.microsoft.com/azureml/curated/azureml-automl-dnn-vision-gpu:131

WORKDIR /app

# ENV QUART_APP=app.py
# ENV QUART_ENV=development
# ENV PYTHONUNBUFFERED True
# ENV PYTHONPATH=/app

COPY ./requirements.txt .
COPY ./req.txt .
COPY conda_env_v_1_0_0.yml .
# COPY scoring_file_v_1_0_0.py .
# COPY model.pt .

# RUN pip3 install torch==1.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install --no-cache-dir -r req.txt
RUN conda env update -f conda_env_v_1_0_0.yml

COPY . ./

# CMD hypercorn -b :$PORT app:app
CMD ls && /bin/bash
