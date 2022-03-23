# SDOMLv2

This gitlab repository contains python scripts for getting and processing SDO/AIA, SDO/HMI, and SDO/EVE files to new cloud-friendly .zarr files (SDOMLv2). The data is essentially the same as previous SDOML. For more details about the SDOML dataset, please refer to the following paper:

"A Machine-learning Data Set Prepared from the NASA Solar Dynamics Observatory Mission" http://adsabs.harvard.edu/abs/2019ApJS..242....7G

The major changes in SDOMLv2 include:

- Using the cloud-friendly Zarr format for the storage of chunked, compressed, N-dimensional arrays (https://zarr.readthedocs.io/en/stable/).
- All fits header information is saved in the meta data.

A brief description about each script:
* aia_fits_to_zarr.py for processing the AIA synoptic fits data to zarr files
* hmi_fits_to_zarr.py for processing the HMI fits data to zarr files
* eve_npy_to_zarr.py for processing the EVE npy files (from SDOMLv1) to zarr files
* save_filelists.py for saving file lists of AIA channels
