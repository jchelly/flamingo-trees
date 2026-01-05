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

from virgo.util.match import match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort


def read_soap_halos_for_snapshot(soap_format, snap_nr, datasets):
    """
    Read the specified properties of SOAP halos at snapshot snap_nr
    """
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


def read_hbt_descendants(hbt_dir, snap_nr, soap):
    """
    Read descendant info for halos at snapshot snap_nr, which requires
    reading the HBT output for snap_nr+1.
    """
    # Determine number of files
    filename = f"{hbt_dir}/{snap_nr+1:03d}/SubSnap_{snap_nr+1:03d}.0.hdf5"
    if comm_rank == 0:
        with h5py.File(filename, "r") as infile:
            nr_files = int(infile["NumberOfFiles"][0])
    else:
        nr_files = None
    nr_files = comm.bcast(nr_files)

    # Read the subhalo data
    filenames = f"{hbt_dir}/{snap_nr+1:03d}/SubSnap_{snap_nr+1:03d}"+".{file_nr}.hdf5"
    mf = phdf5.MultiFile(filenames, file_idx=np.arange(nr_files), comm=comm)
    subhalos = mf.read("Subhalos")

    # Extract quantities we need
    trackid = np.ascontiguousarray(subhalos["TrackId"])
    sinktrackid = np.ascontiguousarray(subhalos["SinkTrackId"])
    snapshotindexofsink = np.ascontiguousarray(subhalos["SnapshotIndexOfSink"])
    descendanttrackid = np.ascontiguousarray(subhalos["DescendantTrackId"])
    del subhalos

    # For each SOAP halo at snap snap_nr, find the same trackid in HBT at snap snap_nr+1
    hbt_index = psort.parallel_match(soap["InputHalos/HBTplus/TrackId"], trackid, comm=comm)
    assert np.all(hbt_index >= 0) # All should match

    # Fetch the matching values
    soap["NextSnapshot/SinkTrackId"] = psort.fetch_elements(sinktrackid, hbt_index, comm=comm)
    soap["NextSnapshot/SnapshotIndexOfSink"] = psort.fetch_elements(snapshotindexofsink, hbt_index, comm=comm)
    soap["NextSnapshot/DescendantTrackId"] = psort.fetch_elements(descendanttrackid, hbt_index, comm=comm)


def make_soap_trees(hbt_dir, soap_format, first_snap, last_snap, output_file, pass_through):

    # Certain datasets are always required from SOAP
    always_read = (
        "InputHalos/HBTplus/TrackId",
        "InputHalos/HBTplus/LastMaxMass",
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
                nr_halos_total += soap["InputHalos/HBTplus/TrackId"].shape[0]
        print(f"Total number of halos: {nr_halos_total}")
    else:
        nr_halos_total = None
    nr_halos_total = comm.bcast(nr_halos_total)

    # Now read in all halo properties. For each property we have a list.
    # Each list item is the local part of a distributed array with the data
    # for one snapshot.
    snapshot = {}
    for snap_nr in range(last_snap, first_snap-1, -1):

        # Read in the halos for this snapshot from SOAP
        snapshot[snap_nr] = read_soap_halos_for_snapshot(soap_format, snap_nr, datasets)

        # Read in the descendant information for this snapshot from HBT
        if snap_nr < last_snap:
            read_hbt_descendants(hbt_dir, snap_nr, snapshot[snap_nr])

        if snap_nr == last_snap:
            # At the final snapshot, repartition for even(ish) load balancing.
            local_nr_halos = len(snapshot[snap_nr]["InputHalos/HBTplus/TrackId"])
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
            descid1  = snapshot[snap_nr]["NextSnapshot/DescendantTrackId"]
            trackid2 = snapshot[snap_nr+1]["InputHalos/HBTplus/TrackId"]
            same_trackid_index = psort.parallel_match(trackid1, trackid2, comm=comm)
            desc_trackid_index = psort.parallel_match(descid1,  trackid2, comm=comm)
            # If (and only if) DescendantTrackId >= 0, we should find the halo with that TrackId at the next snapshot
            assert np.all((desc_trackid_index >= 0) == (snapshot[snap_nr]["NextSnapshot/DescendantTrackId"] >= 0))
            # Choose the descendant track and overwrite DescendantTrackId with the new choice
            snapshot[snap_nr]["NextSnapshot/DescendantTrackId"] = np.where(same_trackid_index >= 0, trackid1, descid1)
            del descid1
            chosen_trackid_index = np.where(same_trackid_index >= 0, same_trackid_index, desc_trackid_index)
            # Determine which MPI rank contains the descendant of each halo
            have_descendant = chosen_trackid_index >= 0
            this_rank = np.ones(len(trackid2), dtype=int)*comm_rank # distributed array with MPI rank of each halo at snapshot snap_nr+1
            destination_rank = np.ones(len(trackid1), dtype=int) * comm_rank # leave halo on current rank if it has no descendant
            destination_rank[have_descendant] = psort.fetch_elements(this_rank, chosen_trackid_index[have_descendant], comm=comm)
            # Count halos to go to each rank
            nr_sent_to_rank = comm.allreduce(np.bincount(destination_rank, minlength=comm_size))
            # Get sorting order by destination
            order = psort.parallel_sort(destination_rank, return_index=True, comm=comm)
            destination_rank = psort.repartition(destination_rank, nr_sent_to_rank, comm=comm)
            assert np.all(destination_rank == comm_rank)
            del destination_rank
            # Rearrange the data
            for name in snapshot[snap_nr]:
                snapshot[snap_nr][name] = psort.fetch_elements(snapshot[snap_nr][name], order, comm=comm)
                snapshot[snap_nr][name] = psort.repartition(snapshot[snap_nr][name], nr_sent_to_rank, comm=comm)

        # Assign unique identifiers to the halos
        trackid1 = snapshot[snap_nr]["InputHalos/HBTplus/TrackId"]
        assert np.all(trackid1 < (1 << 32))
        snapshot[snap_nr]["UniqueId"] = trackid1.astype(np.int64) + (snap_nr << 32)

        # Assign unique descendant identifiers to the halos. At this point, the descendant should
        # always be on the same MPI rank if it exists.
        if snap_nr == last_snap:
            # Halos at the final snapshot have no descendant
            snapshot[snap_nr]["UniqueDescendantId"] = -np.ones_like(snapshot[snap_nr]["UniqueId"])
        else:
            # Halos at earlier snapshots might have a descendant at the next snapshot
            snapshot[snap_nr]["UniqueDescendantId"] = np.where(
                snapshot[snap_nr]["NextSnapshot/DescendantTrackId"] >= 0,
                snapshot[snap_nr]["NextSnapshot/DescendantTrackId"] + ((snap_nr+1) << 32),
                -1)
            # Halos with a descendant ID should all match up now
            local_descendant_index = match(snapshot[snap_nr]["UniqueDescendantId"],
                                           snapshot[snap_nr+1]["UniqueId"])
            assert np.all((local_descendant_index >= 0) | (snapshot[snap_nr]["UniqueDescendantId"] < 0))


if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Make merger trees from SOAP outputs.")
    parser.add_argument("hbt_dir", type=str, help="Directory with HBT-HERONS output")
    parser.add_argument("soap_format", type=str, help="Format string to make SOAP filenames")
    parser.add_argument("first_snap", type=int, help="Index of the first snapshot in the simulation")
    parser.add_argument("last_snap", type=int, help="Index of the last snapshot in the simulation")
    parser.add_argument("output_file", type=str, help="Name of the output file to create")
    parser.add_argument("--pass-through", type=str, default=None, help="Comma separated list of datasets to pass through")
    args = parser.parse_args()

    make_soap_trees(args.hbt_dir, args.soap_format, args.first_snap, args.last_snap, args.output_file, args.pass_through)
