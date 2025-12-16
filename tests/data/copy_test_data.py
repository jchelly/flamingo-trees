#!/bin/env python
#
# Extract datasets we need from a small test run
#

import h5py

simdir="/cosma8/data/dp004/flamingo/Runs/L0200N0360/HYDRO_FIDUCIAL/"
first_snap=0
last_snap=8

datasets = (
    "InputHalos/HBTplus/TrackId",
    "InputHalos/HBTplus/DescendantTrackId",
    "InputHalos/HBTplus/LastMaxMass",
    "InputHalos/NumberOfBoundParticles",
)

for snap_nr in range(first_snap, last_snap+1):

    input_filename  = f"{simdir}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
    output_filename = f"./halo_properties_{snap_nr:04d}.hdf5"

    print(input_filename)
    with (h5py.File(input_filename, "r") as infile,
          h5py.File(output_filename, "w") as outfile):
        outfile.require_group("InputHalos/HBTplus")
        for name in datasets:
            data = infile[name]
            chunks = list(data.shape)
            chunks[0] = min(data.shape[0], 10*1024)
            outfile.create_dataset(name, data=data, chunks=tuple(chunks), shuffle=True, compression="gzip", compression_opts=9)

