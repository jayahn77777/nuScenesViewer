from argparse import ArgumentParser
from pathlib import Path
from typing import List

from tqdm import tqdm
from src.datamodule.nusc_extractor import NuscExtractor
#from src.datamodule.nusc_extractor_multiagent import NuscExtractorMultiAgent


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.json"))
    return scenario_files


def preprocess_batch(extractor, file_list: List[Path]):
    for file in file_list:
        extractor.save(file)


def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        mode_path = data_root / mode / "v1.0-mini"  # 올바른 경로 설정
        if args.multiagent:
            save_dir = data_root / "nusc_multiagent-baseline" / mode
            #extractor = NuscExtractorMultiAgent(input_path=data_root, save_path=save_dir, mode=mode)
            extractor = NuscExtractor(input_path=mode_path, save_path=save_dir, mode=mode)
        else:
            save_dir = data_root / "nusc_forecast-mae" / mode
            extractor = NuscExtractor(input_path=mode_path, save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            for i in range(0, len(scenario_files), batch):
                preprocess_batch(extractor, scenario_files[i:i + batch])
        else:
            for file in tqdm(scenario_files):
                extractor.process()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, required=True)
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--multiagent", "-m", action="store_true")

    args = parser.parse_args()
    preprocess(args)
