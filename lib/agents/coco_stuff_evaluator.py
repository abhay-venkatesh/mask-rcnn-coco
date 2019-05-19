from lib.datasets.coco import CocoDataset
from lib.agents.agent import Agent
from lib.mask_rcnn.model import MaskRCNN


class COCOStuffEvaluator(Agent):
    YEAR = 2017
    LEN_DATASET = 21000

    def run(self):
        dataset_val = CocoDataset()
        val_type = "val" if self.YEAR in '2017' else "minival"
        coco = dataset_val.load_coco(
            self.config["dataset path"],
            val_type,
            year=self.YEAR,
            return_coco=True,
            auto_download=False)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(self.LEN_DATASET))
        net = MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        evaluate_coco(net, dataset_val, coco, "bbox", limit=self.LEN_DATASET)
