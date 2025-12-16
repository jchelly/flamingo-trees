#!/bin/env python
#
# Code to generate depth first indexed merger trees from SOAP output
#

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort


def make_soap_trees(soap_format, first_snap, last_snap, output_file, pass_through):

    # Certain datasets are always required
    always_read = (
        "InputHalos/HBTplus/TrackId",
        "InputHalos/HBTplus/DescendantTrackId",
        "InputHalos/HBTplus/LastMaxMass",
        "InputHalos/NumberOfBoundParticles",
        )

    # Determine datasets we need to read
    datasets = []
    if pass_through is not None:
        datasets += [pt.strip() for pt in pass_through.split(",")]
    for name in always_read:
        datasets.append(name)

    # Determine total number of halos over all snapshots
    if comm_rank == 0:
        nr_halos_total = 0
        for snap_nr in range(first_snap, last_snap+1):
            with h5py.File(soap_format.format(snap_nr=snap_nr), "r") as soap:
                nr_halos_total += soap["InputHalos/NumberOfBoundParticles"].shape[0]
        print(f"Total number of halos: {nr_halos_total}")
    else:
        nr_halos_total = None
    nr_halos_total = comm.bcast(nr_halos_total)

    # Now read in all halo properties. Will create one big array for each one.
    data = {}
    offset = 0
    for snap_nr in range(first_snap, last_snap+1):
        if comm_rank == 0:
            print(f"Reading snapshot: {snap_nr}")
        with h5py.File(soap_format.format(snap_nr=snap_nr), "r", driver="mpio", comm=comm) as soap:
            # Read all properties from SOAP
            for name in datasets:
                # Get metadata for this property
                dataset = soap[name]
                dtype = dataset.dtype
                shape = (nr_halos_total,)+dataset.shape[1:]
                # Allocate storage, if we didn't already
                if name not in data:
                    data[name] = np.ndarray(shape, dtype=dtype)
                n = dataset.shape[0]
                # Read the data
                if comm_rank == 0:
                    print(f"  {name}")
                data[name][offset:offset+n,...] = phdf5.collective_read(dataset, comm)
            # Also store the snapshot number
            if "SnapshotNumber" not in data:
                data["SnapshotNumber"] = np.ndarray(nr_halos_total, dtype=np.int32)
            data["SnapshotNumber"][offset:offset+n] = snap_nr
            # And the index in the SOAP file
            if "SOAPIndex" not in data:
                data["SOAPIndex"] = np.ndarray(nr_halos_total, dtype=np.int32)
            data["SOAPIndex"][offset:offset+n] = np.arange(n, dtype=int)
            # Then advance to the next part of the output arrays
            offset += n


if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Make merger trees from SOAP outputs.")
    parser.add_argument("soap_format", type=str, help="Format string to make SOAP filenames")
    parser.add_argument("first_snap", type=int, help="Index of the first snapshot in the simulation")
    parser.add_argument("last_snap", type=int, help="Index of the last snapshot in the simulation")
    parser.add_argument("output_file", type=str, help="Name of the output file to create")
    parser.add_argument("--pass-through", type=str, default=None, help="Comma separated list of datasets to pass through")
    args = parser.parse_args()

    make_soap_trees(args.soap_format, args.first_snap, args.last_snap, args.output_file, args.pass_through)
