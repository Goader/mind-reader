from typing import Iterable
from pathlib import Path
import re

from omegaconf import DictConfig
import nibabel as nib

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities import rank_zero_info


class NSDDataset(Dataset):
    def __init__(
            self,
            timeseries_files: list[Path],
            design_files: list[Path],
            embeddings_file: Path
    ):
        super().__init__()
        self.timeseries_files = timeseries_files
        self.design_files = design_files
        self.embeddings_file = embeddings_file

        assert len(self.timeseries_files) == len(self.design_files), \
            f'number of timeseries files ({len(self.timeseries_files)}) ' \
            f'does not match the number of design files ({len(self.design_files)})'

        self.embeddings = torch.load(self.embeddings_file, map_location='cpu')

    def __len__(self) -> int:
        return len(self.timeseries_files)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timeseries_file = self.timeseries_files[index]
        design_file = self.design_files[index]

        # Load timeseries data
        timeseries_nib = nib.load(timeseries_file)
        timeseries = torch.tensor(timeseries_nib.get_fdata(), dtype=torch.float32, device='cpu')

        with open(design_file, 'r') as f:
            # We add a zero to the beginning of the design indices to account for the first timepoint
            # which is not included in the design file, despite the documentation saying otherwise.
            #   > Each file is a column vector of integers, and the number of elements corresponds to the
            #   > number of volumes in the functional data preparation for a given run.
            # https://cvnlab.slite.page/p/vjWTghPTb3/Time-series-data
            design_indices = [0] + [int(line.strip()) for line in f.readlines() if line.strip()]
            assert len(design_indices) == timeseries.size(0), \
                f'number of design indices ({len(design_indices)}) ' \
                f'does not match the number of timepoints ({timeseries.size(0)})'
            labels_mask = torch.tensor(design_indices, dtype=torch.bool, device='cpu')
            embeddings = self.embeddings[[index - 1 for index in design_indices]]

        return timeseries, embeddings, labels_mask


class NSDDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_dataset: NSDDataset | None = None
        self.val_dataset: NSDDataset | None = None
        self.test_dataset: NSDDataset | None = None

    def prepare_data(self) -> None:
        pass

    def _iterate_aligned_files(self, timeseries_dir: str, design_dir: str) -> Iterable[tuple[Path, Path]]:
        """Iterate over aligned files in the dataset."""
        timeseries_dir = Path(timeseries_dir)
        design_dir = Path(design_dir)
        for subject in timeseries_dir.iterdir():
            for timeseries_file in subject.iterdir():
                session_id = re.search(r"session(\d+)", timeseries_file.name).group(1)
                run_id = re.search(r"run(\d+)", timeseries_file.name).group(1)

                design_file = design_dir / subject.name / f"design_session{session_id}_run{run_id}.tsv"

                if not design_file.exists():
                    rank_zero_info(f'Warning: design file {design_file} does not exist. Skipping.')
                    continue
                yield timeseries_file, design_file

    def setup(self, stage: str | None = None) -> None:
        """Prepare the dataset for training and evaluation."""
        self.train_dataset = NSDDataset(
            *zip(*self._iterate_aligned_files(self.cfg.datamodule.timeseries.train,
                                              self.cfg.datamodule.design.train)),
            self.cfg.datamodule.embeddings,
        )
        self.val_dataset = NSDDataset(
            *zip(*self._iterate_aligned_files(self.cfg.datamodule.timeseries.val,
                                              self.cfg.datamodule.design.val)),
            self.cfg.datamodule.embeddings,
        )
        self.test_dataset = NSDDataset(
            *zip(*self._iterate_aligned_files(self.cfg.datamodule.timeseries.test,
                                              self.cfg.datamodule.design.test)),
            self.cfg.datamodule.embeddings,
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.cfg.datamodule.batch_size,
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.cfg.datamodule.batch_size,
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.cfg.datamodule.batch_size,
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory)
