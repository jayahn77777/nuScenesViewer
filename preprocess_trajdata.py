from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import ray
from unified_extractor import UnifiedExtractor  # 새로 만든 Extractor 불러오기

ray.init(num_cpus=16)

def preprocess(args):
    data_root = Path(args.data_root)
    save_dir = data_root / "unified-dataset" / args.mode
    extractor = UnifiedExtractor(save_path=save_dir, mode=args.mode)

    save_dir.mkdir(exist_ok=True, parents=True)
    total_samples = len(extractor.dataset)

    for idx in tqdm(range(total_samples)):
        extractor.save(idx)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, required=True)
    parser.add_argument("--mode", "-m", type=str, default="train")

    args = parser.parse_args()
    preprocess(args)