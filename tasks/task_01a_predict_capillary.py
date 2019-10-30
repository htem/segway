import logging
import sys
import daisy
import task_helper2 as task_helper

from task_predict_ilastik import PredictIlastikTask

logger = logging.getLogger(__name__)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': PredictIlastikTask(
                            task_id="PredictCapillaryTask",
                            global_config=global_config,
                            **user_configs),
         'request': req_roi}],
        global_config=global_config)
