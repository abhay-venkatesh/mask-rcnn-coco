from lib.agents.agent import Agent
from lib.datasets.coco import CocoDataset
from lib.mrcnn import coco_eval
from lib.mrcnn.model import MaskRCNN
from pathlib import Path
from tqdm import tqdm
import numpy as np
import simplejson as json


class InferenceConfig(coco_eval.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class COCOStuffValidator(Agent):
    YEAR = 2017

    def run(self):
        dataset_val = CocoDataset()
        val_type = "val"
        inference_config = InferenceConfig()
        coco = dataset_val.load_coco(
            self.config["dataset path"],
            val_type,
            year=self.YEAR,
            return_coco=True,
            auto_download=False)
        dataset_val.prepare()
        net = MaskRCNN(
            mode="inference",
            config=inference_config,
            model_dir=self.config["checkpoints folder"])
        self._validate(net, dataset_val, coco)

    def _validate(self, model, dataset, coco, limit=0, image_ids=None):
        # Pick COCO images from the dataset
        image_ids = image_ids or dataset.image_ids

        # Limit to a subset
        if limit:
            image_ids = image_ids[:limit]

        # Get corresponding COCO image IDs.
        coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

        results = []
        for i, image_id in tqdm(enumerate(image_ids)):
            # Load images
            image = dataset.load_image(image_id)

            # Run detection
            r = model.detect([image], verbose=0)[0]

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            image_results = coco_eval.build_coco_results(
                dataset, coco_image_ids[i:i + 1], r["rois"], r["class_ids"],
                r["scores"], r["masks"].astype(np.uint8))
            image_results = self._cast(image_results)
            results.extend(image_results)
            break

        with open(
                Path(self.config["outputs folder"], "coco_result.json"),
                "w+") as f:
            json.dump(results, f)

    def _cast(self, image_results):
        image_results_ = []
        for result in image_results:
            result_ = result
            result_["bbox"] = [int(i) for i in result["bbox"]]
            result_["score"] = float(result["score"])
            image_results_.append(result_)
        return image_results_
