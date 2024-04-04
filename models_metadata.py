import os
import re
import json

from dotenv import load_dotenv
from azure.ai.ml import MLClient # Input
# from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from custom_exceptions import(
    ModelFilePathPatternError
)

load_dotenv()

NACHET_SUBSCRIPTION_ID = os.getenv("NACHET_SUBSCRIPTION_ID")
NACHET_RESOURCE_GROUP = os.getenv("NACHET_RESOURCE_GROUP")
NACHET_WORKSPACE = os.getenv("NACHET_WORKSPACE")

ml_client = MLClient(
    DefaultAzureCredential(),
    NACHET_SUBSCRIPTION_ID,
    NACHET_RESOURCE_GROUP, NACHET_WORKSPACE)

def generate_model_endpoints_metadata(ml_client) -> list:
    """
    Retrieves deployed online endpoints and returns a list of dictionaries
    containing their metadata
    """

    model_endpoints_metadata = []

    # Retrieve all endpoints containing "nachet"
    endpoints = ml_client.online_endpoints.list()
    nachet_endpoints = [endpoint for endpoint in endpoints if 'nachet' in endpoint.name.lower()]

    for ep in nachet_endpoints:
        model_endpoint_metadata = {
            'endpoint_name': "",
            'model_name': '',
            'created_by': '',
            'creation_date': '',
            'version': '',
            'description': '',
            'job_name': '',
            'dataset': '',
            'metrics': [],
            'identifiable': []  # List of seeds model can identify
        }

        # Retrieve online_deployment
        deployment = ml_client.online_deployments.get(
            endpoint_name=ep.name,
            name=list(ep.traffic.keys())[0])

        # Retrieve deployment's model (from filePath)
        model_filepath = deployment.model
        pattern = re.compile(r"models/([^/]+)/versions/(\d+)")
        match = pattern.search(model_filepath)
        if match:
            model_name = match.group(1)
            model_version = match.group(2)
        else:
            raise ModelFilePathPatternError("Error retrieving model name or version from filePath.")

        # Retrieve the job object from model
        model = ml_client.models.get(name=model_name, version=model_version)
        job = ml_client.jobs.get(name=model.job_name)

        model_endpoint_metadata['endpoint_name'] = ep.name
        model_endpoint_metadata['model_name'] = model_name
        model_endpoint_metadata['created_by'] = job.creation_context.created_by
        model_endpoint_metadata['creation_date'] = job.creation_context.created_at.strftime("%Y-%m-%d")
        model_endpoint_metadata['version'] = model_version
        model_endpoint_metadata['description'] = model.description
        model_endpoint_metadata['job_name'] = job.display_name

        model_endpoints_metadata.append(model_endpoint_metadata)

    return model_endpoints_metadata

if __name__ == "__main__":
    model_endpoints_metadata = generate_model_endpoints_metadata(ml_client)
    with open("model_endpoints_metadata.json", "w") as outfile:
        json.dump(model_endpoints_metadata, outfile, indent=4)
