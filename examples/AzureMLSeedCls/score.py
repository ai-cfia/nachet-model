import os
import logging
from transformers import pipeline
# import request


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # print(os.listdir(os.getenv("AZUREML_MODEL_DIR")),'DIR PATHHHHHHHHHH LISTTTTTTTTT','\n\n\n')
    # model_path = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"),'checkpoint-1500'
    # )
    print(os.listdir("/app/artifacts/"), "\n\n\n")
    model_path = "/app/artifacts/SwinV1_Base_DataAugv2_"
    # deserialize the model file back into a sklearn model
    # model = joblib.load(model_path)
    # model = SwinForImageClassification.from_pretrained(pth)
    model = pipeline(model=model_path, task="image-classification")
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    # print(raw_data,'RAWWW DATAAAAA','\n\n\n')
    # model.eval()
    # image = Image.open(raw_data)

    logging.info("model 1: request received")
    # data = json.loads(raw_data)["data"]
    # data = numpy.array(data)
    results = model(raw_data)
    logging.info("Request processed")
    return results
