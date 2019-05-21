from lib.agents.agent import Agent
from lib.datasets.coco import CocoDataset
from lib.mrcnn import coco_eval
from lib.mrcnn.model import MaskRCNN
from tqdm import tqdm
import numpy as np
import time


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

        t_prediction = 0
        t_start = time.time()

        results = []
        for i, image_id in tqdm(enumerate(image_ids)):
            # Load images
            image = dataset.load_image(image_id)

            # Run detection
            t = time.time()
            r = model.detect([image], verbose=0)[0]
            t_prediction += (time.time() - t)

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            image_results = coco_eval.build_coco_results(
                dataset, coco_image_ids[i:i + 1], r["rois"], r["class_ids"],
                r["scores"], r["masks"].astype(np.uint8))
            print(image_results)
            results.extend(image_results)

        # Load results. This modifies results with additional attributes.
        coco_results = coco.loadRes(results)

        print("Prediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / len(image_ids)))

        print("Total time: ", time.time() - t_start)

        return coco_results
