from lib.agents.agent import Agent


class COCOStuffEvaluator(Agent):
    def run(self):
        raise NotImplementedError
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(
            args.dataset,
            val_type,
            year=args.year,
            return_coco=True,
            auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit)