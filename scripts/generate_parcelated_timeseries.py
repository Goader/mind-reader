from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--timeseries', type=str, nargs='+', help='original timeseries data')
    parser.add_argument('--roi', type=str, nargs='+', help='roi map')
    parser.add_argument('--output_dir', type=str, help='aggregated timeseries data')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for timeseries_filepath, roi_filepath in zip(args.timeseries, args.roi):
        timeseries = nib.load(timeseries_filepath)
        roi = nib.load(roi_filepath)

        X = timeseries.get_fdata()
        mask = roi.get_fdata()

        seq_len = X.shape[-1]
        components_count = np.max(mask) - 1

        Y = np.empty((seq_len, components_count), dtype=np.float32)

        for component_idx in range(1, components_count):
            region_mask = mask == component_idx
            Y[:, component_idx - 1] = np.mean(X[region_mask], axis=(0, 1, 2))

        filename = Path(timeseries_filepath).name.split('.')[0] + '.nii.gz'
        output_filepath = output_dir / filename
        nib.save(nib.Nifti1Image(Y, timeseries.affine, timeseries.header), output_filepath)
