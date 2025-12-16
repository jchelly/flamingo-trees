#!/bin/env python
#
# Extract datasets we need from a small test run
#

import h5py

simdir="/cosma8/data/dp004/flamingo/Runs/L0200N0360/HYDRO_FIDUCIAL/"
first_snap=0
last_snap=8

for snap_nr in range(first_snap, last_snap+1):

    input_filename  = f"{simdir}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
    output_filename = f"./halo_properties_{snap_nr:04d}.hdf5"

    print(input_filename)
    with (h5py.File(input_filename, "r") as infile,
          h5py.File(output_filename, "w") as outfile):
          in_group  = infile["InputHalos/HBTplus"]
          out_group = outfile.require_group("InputHalos/HBTplus")
          trackid = in_group["TrackId"][...]
          chunks = (min(len(trackid), 10*1024),)
          out_group.create_dataset("TrackId", data=trackid, chunks=chunks, shuffle=True, compression="gzip", compression_opts=9)

