#!/bin/env python
#
# Extract datasets we need from a small test run
#

import h5py
import os
import numpy as np

simdir="/cosma8/data/dp004/flamingo/Runs/L0200N0360/HYDRO_FIDUCIAL/"
first_snap=0
last_snap=8

datasets = (
    "InputHalos/HBTplus/TrackId",
    "InputHalos/HBTplus/LastMaxMass",
)

for snap_nr in range(first_snap, last_snap+1):

    print(f"Snapshot {snap_nr}")

    # Copy SOAP data
    input_filename  = f"{simdir}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
    output_filename = f"./halo_properties_{snap_nr:04d}.hdf5"
    with (h5py.File(input_filename, "r") as infile,
          h5py.File(output_filename, "w") as outfile):
        outfile.require_group("InputHalos/HBTplus")
        for name in datasets:
            data = infile[name]
            chunks = list(data.shape)
            chunks[0] = min(data.shape[0], 10*1024)
            outfile.create_dataset(name, data=data, chunks=tuple(chunks), shuffle=True, compression="gzip", compression_opts=9)

    # Copy parts of HBT output which we need
    os.makedirs(f"./HBT/{snap_nr:03d}", exist_ok=True)
    file_nr = 0
    nr_files = 1
    while file_nr < nr_files:
        input_filename = f"{simdir}/HBT/{snap_nr:03d}/SubSnap_{snap_nr:03d}.{file_nr}.hdf5"
        output_filename = f"./HBT/{snap_nr:03d}/SubSnap_{snap_nr:03d}.{file_nr}.hdf5"
        with h5py.File(input_filename, "r") as infile:
            subhalos_in = infile["Subhalos"][...]
            nr_files = int(infile["NumberOfFiles"][0])
            total_nr_subhalos = int(infile["NumberOfSubhalosInAllFiles"][0])
        dtype = [
            ("TrackId",             subhalos_in["TrackId"].dtype),
            ("SinkTrackId",         subhalos_in["SinkTrackId"].dtype),
            ("SnapshotIndexOfSink", subhalos_in["SnapshotIndexOfSink"].dtype),
            ("DescendantTrackId",   subhalos_in["DescendantTrackId"].dtype),
        ]
        subhalos_out = np.ndarray(len(subhalos_in), dtype=dtype)
        for name in subhalos_out.dtype.fields.keys():
            subhalos_out[name] = subhalos_in[name]
        chunks = (min(len(subhalos_in), 10*1024),)
        with h5py.File(output_filename, "w") as outfile:
            outfile.create_dataset("Subhalos", data=subhalos_out, chunks=chunks, shuffle=True, compression="gzip", compression_opts=9)
            outfile["NumberOfFiles"] = (nr_files,)
            outfile["NumberOfSubhalosInAllFiles"] = (total_nr_subhalos,)
        file_nr += 1
