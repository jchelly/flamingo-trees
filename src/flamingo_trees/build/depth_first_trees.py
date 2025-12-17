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


def read_halos_for_snapshot(soap_format, snap_nr, datasets):

    if comm_rank == 0:
        print(f"Reading snapshot: {snap_nr}")
    data = {}
    with h5py.File(soap_format.format(snap_nr=snap_nr), "r", driver="mpio", comm=comm) as soap:
        # Read all properties from SOAP
        for name in datasets:
            data[name] = phdf5.collective_read(soap[name], comm)
            # Store number of halos read from this snapshot on this MPI rank
            nr_halos_local = data["InputHalos/HBTplus/TrackId"].shape[0]
            # Store the snapshot number for these halos
            data["SnapshotNumber"] = np.ones(nr_halos_local, dtype=np.int32)*snap_nr
            # Store the index in the SOAP catalogue for these halos
            nr_halos_prev = comm.scan(nr_halos_local) - nr_halos_local
            data["SOAPIndex"] = np.arange(nr_halos_local, dtype=int)+nr_halos_prev
    return data


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
    snapshot = {}
    for snap_nr in range(last_snap, first_snap-1, -1):

        # Read in the halos for this snapshot
        snapshot[snap_nr] = read_halos_for_snapshot(soap_format, snap_nr, datasets)

        if snap_nr == last_snap:
            # At the final snapshot, repartition to have a fixed number of
            # halos per rank with any extra on the last rank. This makes it
            # easy to compute what rank a halo is on given its global index.
            local_nr_halos = len(snapshot[snap_nr]["InputHalos/NumberOfBoundParticles"])
            total_nr_halos = comm.allreduce(local_nr_halos)
            nr_halos_per_rank = total_nr_halos // comm_size
            nr_sent_to_rank = np.ones(comm_size, dtype=int)*nr_halos_per_rank
            nr_sent_to_rank[-1] = total_nr_halos - np.sum(nr_sent_to_rank[:-1])
            assert np.sum(nr_sent_to_rank) == total_nr_halos
            for name in snapshot[snap_nr]:
                snapshot[snap_nr][name] = psort.repartition(snapshot[snap_nr][name], nr_sent_to_rank, comm=comm)
        else:
            # At earlier times we want to send each halo to whatever MPI rank contains its descendant.
            # Need to define the descendant first: if the same TrackId exists in SOAP at the next
            # snapshot, then that is the descendant. Otherwise we go with DescendantTrackId.
            # First, find the global index of any matching TrackId at the next snapshot.
            trackid1 = snapshot[snap_nr]["InputHalos/HBTplus/TrackId"]
            descid1  = snapshot[snap_nr]["InputHalos/HBTplus/DescendantTrackId"]
            trackid2 = snapshot[snap_nr+1]["InputHalos/HBTplus/TrackId"]
            same_trackid_index = psort.parallel_match(trackid1, trackid2, comm=comm)
            desc_trackid_index = psort.parallel_match(descid1,  trackid2, comm=comm)
            destination_rank = np.where(same_trackid_index >= 0,
                                        np.clip(same_trackid_index // nr_halos_per_rank, a_min=0, a_max=comm_size-1),
                                        np.clip(desc_trackid_index // nr_halos_per_rank, a_min=0, a_max=comm_size-1))
            # Some subhalos might have no descendant at all. Leave these on their current rank so
            # that they follow SOAP's spatial ordering and are spread over all ranks.
            no_descendant = (same_trackid_index < 0) & (desc_trackid_index < 0)
            destination_rank[no_descendant] = comm_rank
            # Count halos to go to each rank
            nr_sent_to_rank = comm.allreduce(np.bincount(destination_rank, minlength=comm_size))
            # Get sorting order by destination
            order = psort.parallel_sort(destination_rank, return_index=True, comm=comm)
            del destination_rank
            # Rearrange the data
            for name in snapshot[snap_nr]:
                snapshot[snap_nr][name] = psort.fetch_elements(snapshot[snap_nr][name], order, comm=comm)
                snapshot[snap_nr][name] = psort.repartition(snapshot[snap_nr][name], nr_sent_to_rank, comm=comm)


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
