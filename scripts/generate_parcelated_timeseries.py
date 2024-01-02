from argparse import ArgumentParser
from pathlib import Path
import glob
import tqdm

import nibabel as nib
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--timeseries', type=str, nargs='+', help='original timeseries data glob patterns')
    parser.add_argument('--roi', type=str, help='roi map filepath')
    parser.add_argument('--output_dir', type=str, help='aggregated timeseries data')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    timeseries_filepaths = [
        timeseries_filepath
        for timeseries_glob in args.timeseries
        for timeseries_filepath in glob.glob(timeseries_glob)
    ]

    roi = nib.load(args.roi)
    for timeseries_filepath in tqdm.tqdm(timeseries_filepaths):
        timeseries = nib.load(timeseries_filepath)

        X = timeseries.get_fdata()
        mask = roi.get_fdata()

        seq_len = X.shape[-1]
        components_count = int(np.max(mask))

        Y = np.empty((seq_len, components_count), dtype=np.float32)

        for component_idx in range(1, components_count + 1):
            region_mask = mask == component_idx
            Y[:, component_idx - 1] = np.mean(X[region_mask], axis=0)

        filename = Path(timeseries_filepath).name.split('.')[0] + '.nii.gz'
        output_filepath = output_dir / filename
        nib.save(nib.Nifti1Image(Y, timeseries.affine, timeseries.header), output_filepath)
