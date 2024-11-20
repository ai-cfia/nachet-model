FROM mcr.microsoft.com/azureml/curated/azureml-automl-dnn-vision-gpu:131

WORKDIR /app

ENV AZUREML_MODEL_DIR=/app/artifacts

COPY ./requirements.txt .
COPY ./req.txt .
COPY ./conda_env_v_1_0_0.yml .
# COPY scoring_file_v_1_0_0.py .
# COPY model.pt .

# RUN pip3 install torch==1.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install --no-cache-dir -r req.txt
RUN conda env update -f conda_env_v_1_0_0.yml

COPY . /project/
COPY . ./

RUN chown -R 1000:1000 /app
RUN chmod +x /app/startup.sh
# RUN chmod -R 777 /app

# RUN chown -R 1000:1000 /project
# RUN chmod +x /project/startup.sh
# RUN chmod -R 777 /project

EXPOSE 5001
