from argparse import ArgumentParser
from pathlib import Path
from typing import List

import ray
from tqdm import tqdm

from src.datamodule.av2_extractor import Av2Extractor
from src.datamodule.av2_extractor_multiagent import Av2ExtractorMultiAgent
from src.datamodule.nuscenes_extractor import NuScenesExtractor
from src.utils.ray_utils import ActorHandle, ProgressBar

ray.init(num_cpus=16)

def glob_files(data_root: Path, mode: str, dataset_type: str):
    file_root = data_root / mode
    if dataset_type == "argoverse":
        scenario_files = list(file_root.rglob("*.parquet"))
    elif dataset_type == "nuscenes":
        scenario_files = [scene["token"] for scene in NuScenesExtractor().nusc.scene]
    else:
        raise ValueError("Unsupported dataset type")
    return scenario_files

@ray.remote
def preprocess_batch(extractor, file_list: List[Path], pb: ActorHandle):
    for file in file_list:
        extractor.save(file)
        pb.update.remote(1)

def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        if args.dataset_type == "argoverse":
            if args.multiagent:
                save_dir = data_root / "multiagent-baseline" / mode
                extractor = Av2ExtractorMultiAgent(save_path=save_dir, mode=mode)
            else:
                save_dir = data_root / "forecast-mae" / mode
                extractor = Av2Extractor(save_path=save_dir, mode=mode)
        elif args.dataset_type == "nuscenes":
            save_dir = data_root / "nuscenes-baseline" / mode
            extractor = NuScenesExtractor(save_path=save_dir, mode=mode, dataset_path=args.data_root)
        else:
            raise ValueError("Unsupported dataset type")

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode, args.dataset_type)

        if args.parallel:
            pb = ProgressBar(len(scenario_files), f"preprocess {mode}-set")
            pb_actor = pb.actor

            for i in range(0, len(scenario_files), batch):
                preprocess_batch.remote(
                    extractor, scenario_files[i : i + batch], pb_actor
                )

            pb.print_until_done()
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, required=True)
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--multiagent", "-m", action="store_true")
    parser.add_argument("--dataset_type", "-t", type=str, required=True, choices=["argoverse", "nuscenes"])

    args = parser.parse_args()
    preprocess(args)
