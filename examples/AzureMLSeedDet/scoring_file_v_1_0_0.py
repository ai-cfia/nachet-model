# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
import json

from azureml.automl.core.shared import logging_utilities
from azureml.automl.dnn.vision.object_detection.writers.score import _score_with_model
from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.common.model_export_utils import (
    load_model,
    run_inference_helper,
)
from azureml.automl.dnn.vision.common.utils import _set_logging_parameters
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from azureml.core.model import Model

TASK_TYPE = "image-object-detection"
logger = get_logger("azureml.automl.core.scoring_script_images")


def init():
    global model

    # Set up logging
    _set_logging_parameters(TASK_TYPE, {})

    # print contents of azure model dir

    model_path = "/app/artifacts/model.pt"
    # model_path = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"),'Seed_Detector_RCNN', 'model.pt'
    # )
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pt')
    if not os.path.exists(model_path):
        model_path = Model.get_model_path(model_name="Seed_Detector_RCNN")
        raise FileNotFoundError("Model not found at path: {}".format(model_path))

    try:
        logger.info("Loading model from path: {}.".format(model_path))
        model_settings = {
            "min_size": 600,
            "max_size": 1333,
            "box_score_thresh": 0.3,
            "nms_iou_thresh": 0.5,
            "box_detections_per_img": 100,
            "tile_grid_size": None,
            "tile_overlap_ratio": 0.25,
            "tile_predictions_nms_thresh": 0.25,
        }
        model = load_model(TASK_TYPE, model_path, **model_settings)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@rawhttp
def run(request):
    logger.info("Request: [{0}]".format(request))
    # print("Request: [{0}]".format(request))
    if request.method == "GET":
        response_body = str.encode(request.full_path)
        return AMLResponse(response_body, 200)

    elif request.method == "POST":
        data = {}
        # request_body = request.get_data()
        logger.info("Running inference.")
        data["image"] = request.get_json()["input_data"]["data"][0]
        result = run_inference_helper(
            model, json.dumps(data), _score_with_model, TASK_TYPE
        )
        logger.info("Finished inferencing.")
        return AMLResponse(result, 200)
    else:
        return AMLResponse("bad request", 500)
