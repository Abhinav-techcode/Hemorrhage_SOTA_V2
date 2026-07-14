import os

# from glob import glob
import click
import nibabel as nib
import numpy as np
import pandas as pd
import skimage.io as io
from skimage import img_as_bool, img_as_ubyte


@click.command()
@click.option(
    'volume_path',
    '-v',
    type=click.Path(exists=True),
    default="../data/volumes"
)
@click.option(
    'slice_path',
    '-s',
    type=click.Path(exists=False),
    default="../data/slices"
)
@click.option(
    '--verbose/--no_verbose',
    default=True,
    help='Whether to print intermediate informations. Default False.'
)
def main(volume_path, slice_path, verbose):
    """
    Process the Seg-CQ500 Nifti Volumes pairs (CT/mask) and save them as a 2D dataset in .tif/.bmp.

    INPUT
        volume_path -v: path to the directory containing the Nifti volumes and the info.csv.
        slice_path  -s: path defining where to save the extracted 2D dataset.
    """
    # read info.csv
    df_info = pd.read_csv(os.path.join(volume_path, 'info.csv'), index_col=0)

    # for each volume
    out_info = []
    for _, sample in df_info.iterrows():
        if verbose:
            print(f">>> Extracting volume {sample['name']}")
        # read volume and get slice
        vol = np.rot90(nib.load(os.path.join(volume_path, sample.CT_fn)).get_fdata(), axes=(0, 1)).astype(np.int16)
        mask = np.rot90(img_as_bool(nib.load(os.path.join(volume_path, sample.mask_fn)).get_fdata()), axes=(0, 1))

        # check shapes
        if mask.shape != vol.shape:
            print(f">>> Warning! The ct volume {sample['name']} does not have "
                  f"the same dimension as the ground truth. CT ({vol.shape}) vs Mask ({mask.shape})")

        # make sub-directory
        if not os.path.exists(os.path.join(slice_path, f'{sample["name"]}/CT/')):
            os.makedirs(os.path.join(slice_path, f'{sample["name"]}/CT/'))
        if not os.path.exists(os.path.join(slice_path, f'{sample["name"]}/mask/')):
            os.makedirs(os.path.join(slice_path, f'{sample["name"]}/mask/'))

        # for each slice
        for i, slice in enumerate(range(vol.shape[2])):
            # save slice (in .tif)
            slice_fn = f'{sample["name"]}/CT/{slice:03}.tif'
            io.imsave(os.path.join(slice_path, slice_fn), vol[:, :, slice], check_contrast=False)

            # save mask (in .bmp) if any ICH
            if np.any(mask[:, :, slice]):
                mask_fn = f'{sample["name"]}/mask/{slice:03}.bmp'
                io.imsave(os.path.join(slice_path, mask_fn), img_as_ubyte(mask[:, :, slice]), check_contrast=False)
                ICH = 1
            else:
                mask_fn = 'None'
                ICH = 0

            # store slice info in df
            out_info.append(dict(id=sample["name"], slice=slice, CT_fn=slice_fn, mask_fn=mask_fn, ICH=ICH))

            if verbose:
                print(f"Slice {i:03}/{vol.shape[2]:03}", end="\r")
        if verbose:
            print("")

    # save slice info df
    df_slice_info = pd.DataFrame(out_info)
    df_slice_info.to_csv(os.path.join(slice_path, 'info.csv'))


if __name__ == '__main__':
    main()
