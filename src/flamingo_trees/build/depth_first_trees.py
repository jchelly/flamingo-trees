#!/bin/env python
#
# Code to generate depth first indexed merger trees from SOAP output
#

from collections import defaultdict

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

    # Now read in all halo properties. For each property we have a list.
    # Each list item is the local part of a distributed array with the data
    # for one snapshot.
    data = defaultdict(list)
    for snap_nr in range(first_snap, last_snap+1):
        if comm_rank == 0:
            print(f"Reading snapshot: {snap_nr}")
        with h5py.File(soap_format.format(snap_nr=snap_nr), "r", driver="mpio", comm=comm) as soap:
            # Read all properties from SOAP
            for name in datasets:
                data[name].append(phdf5.collective_read(soap[name], comm))
            # Store number of halos read from this snapshot on this MPI rank
            nr_halos_local = data["InputHalos/HBTplus/TrackId"][-1].shape[0]
            # Store the snapshot number for these halos
            data["SnapshotNumber"].append(np.ones(nr_halos_local, dtype=np.int32)*snap_nr)
            # Store the index in the SOAP catalogue for these halos
            nr_halos_prev = comm.scan(nr_halos_local) - nr_halos_local
            data["SOAPIndex"].append(np.arange(nr_halos_local, dtype=int)+nr_halos_prev)

    for snap_nr in range(first_snap, last_snap+1):
        if comm_rank == 0:
            print(f"Assigning IDs for snapshot: {snap_nr}")

        i = snap_nr - first_snap
        # Assign a unique ID to each halo by combining snap num and trackid
        trackid = data["InputHalos/HBTplus/TrackId"][i]
        assert np.all(trackid < (1 << 32))
        snapnum = data["SnapshotNumber"][i]
        data["UniqueId"].append(trackid.astype(np.int64) + (snapnum.astype(np.int64) << 32))

        # Assign a unique descendant ID to each halo
        descendant_trackid = data["InputHalos/HBTplus/DescendantTrackId"][i]
        if snap_nr < last_snap:
            # If the same trackid exists at the next snapshot in SOAP, that's the descendant.
            # Otherwise we go with the descendant trackid.
            later_trackid = data["InputHalos/HBTplus/TrackId"][i + 1]
            same_trackid_index = psort.parallel_match(trackid, later_trackid, comm=comm)
            chosen_descendant_trackid = np.where(same_trackid_index>=0, trackid, descendant_trackid)
            data["UniqueDescendantId"].append(chosen_descendant_trackid.astype(np.int64) + ((snap_nr+1) << 32))
        else:
            # At the last snapshot, no halo has a descendant
            data["UniqueDescendantId"].append(-np.ones_like(trackid, dtype=np.int64))

    # Assign halos with no descendant to ranks? Maybe spatially? Stick with SOAP decomposition?
    # Move halos with descendant to same rank as descendant?


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
