#!/bin/env python
#
# Code to generate index arrays for SOAP output
#

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort


def read_snapshot_data(soap_format, snap_nr, cache={}):

    # Datasets we need from each snapshot
    dataset_names = (
        "InputHalos/HBTplus/TrackId",
    )

    # Discard data we don't need any more
    for i in list(cache.keys()):
        if i < snap_nr - 1:
            if comm_rank == 0:
                print(f"  Discarding snapshot {i}")
            del cache[i]

    # Ensure this snapshot is in the cache
    if snap_nr not in cache:
        if comm_rank == 0:
            print(f"  Reading snapshot {snap_nr}")
        data = {}
        with h5py.File(soap_format.format(snap_nr=snap_nr), "r", driver="mpio", comm=comm) as soap:
            for name in dataset_names:
                data[name] = phdf5.collective_read(soap[name], comm)
        cache[snap_nr] = data

    # Return the cached data
    return cache[snap_nr]


def make_soap_index(soap_format, first_snap, last_snap, output_file):

    # Create group for per-snapshot info
    snap_group = output_file.create_group("Snapshots")

    for snap_nr in range(first_snap, last_snap+1):

        if comm_rank == 0:
            print(f"Processing snapshot {snap_nr}")

        # Read SOAP data for the current snapshot
        current_snap = read_snapshot_data(soap_format, snap_nr)

        # Find the maximum TrackId over all ranks
        trackids = current_snap["InputHalos/HBTplus/TrackId"]
        if len(trackids) > 0:
            local_max_trackid = np.amax(trackids)
        else:
            local_max_trackid = -1
        max_trackid = comm.allreduce(local_max_trackid, op=MPI.MAX)

        nr_trackids = max_trackid + 1
        if nr_trackids > 0:
            # Make a distributed array with values 0 to max_trackid (inclusive)
            trackids_per_rank = nr_trackids // comm_size
            trackids_this_rank = trackids_per_rank
            if comm_rank == comm_size-1:
                trackids_this_rank = nr_trackids - (comm_size-1)*trackids_per_rank
            assert comm.allreduce(trackids_this_rank) == nr_trackids
            trackids_prev_ranks = trackids_per_rank * comm_rank
            trackids_to_find = np.arange(trackids_this_rank, dtype=trackids.dtype) + trackids_prev_ranks
            # Find each trackid in SOAP
            soap_index_of_trackid = psort.parallel_match(trackids_to_find, trackids, comm=comm)
        else:
            # There are no halos at this snapshot
            soap_index_of_trackid = np.zeros(0, dtype=int)

        # Write out the result
        current_snap_group = snap_group.create_group(f"{snap_nr:04d}")
        dset = phdf5.collective_write(current_snap_group, "SOAPIndexOfTrackId", soap_index_of_trackid, comm, gzip=6)
        dset.attrs["Description"] = "For each TrackId this gives the index in the SOAP catalogue, or -1 if the TrackId is not present"


if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Compute index arrays for SOAP outputs.")
    parser.add_argument("soap_format", type=str, help="Format string to make SOAP filenames")
    parser.add_argument("first_snap", type=int, help="Index of the first snapshot in the simulation")
    parser.add_argument("last_snap", type=int, help="Index of the last snapshot in the simulation")
    parser.add_argument("output_file", type=str, help="Name of the output file to create")
    args = parser.parse_args()

    with h5py.File(args.output_file, "w", driver="mpio", comm=comm) as output_file:
        make_soap_index(args.soap_format, args.first_snap, args.last_snap, output_file)
