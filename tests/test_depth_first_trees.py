#!/bin/env python

import h5py
import numpy as np
import pytest
from virgo.util.match import match

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from flamingo_trees.build.depth_first_trees import make_soap_trees


@pytest.mark.mpi
def test_depth_first_trees(tmp_path):
    """
    Run the SOAP merger tree code and check the output
    """
    # Parameters for test run
    hbt_dir = "./tests/data/HBT/"
    soap_format = "./tests/data/halo_properties_{snap_nr:04d}.hdf5"
    first_snap = 0
    last_snap = 8

    # Different MPI ranks generate different tmp paths!
    # Here everyone will use the one from rank zero.
    tmp_path = comm.bcast(tmp_path)

    # Run the code
    output_file = tmp_path / "merger_tree.hdf5"
    with h5py.File(output_file, "w", driver="mpio", comm=comm) as f:
        make_soap_trees(hbt_dir, soap_format, first_snap, last_snap, f, None)

    # Do checks on one rank for simplicity
    if comm_rank == 0:

        tree = {}
        snapshot = {}
        with h5py.File(output_file, "r") as f:
            # Read in the merger tree arrays
            for name in f["Tree"]:
                tree[name] = f["Tree"][name][...]
            # Read in the snapshot arrays
            for snap_nr in range(first_snap, last_snap+1):
                group = f["Snapshots"][f"{snap_nr:04d}"]
                snapshot[snap_nr] = {}
                for name in group:
                    snapshot[snap_nr][name] = group[name][...]

        # GalaxyId should start at zero and be unique and sorted
        assert np.all(tree["GalaxyId"] == np.arange(len(tree["GalaxyId"]), dtype=int))

        # Check that converting a SOAP index to a tree index and back round trips correctly.
        for snap_nr in range(first_snap, last_snap+1):
            tree_index_of_soap = snapshot[snap_nr]["TreeIndexOfSOAPHalo"]
            soap_index = np.arange(len(tree_index_of_soap), dtype=int)
            assert np.all(tree["SOAPIndex"][tree_index_of_soap] == soap_index)

        # Check that converting a TrackId to a tree index and back also round trips.
        # Some TrackIds aren't in the tree and have TreeIndex=-1.
        for snap_nr in range(first_snap, last_snap+1):
            tree_index_of_trackid = snapshot[snap_nr]["TreeIndexOfTrackId"]
            trackid = np.arange(len(tree_index_of_trackid), dtype=int)
            is_resolved = tree_index_of_trackid >= 0
            assert np.all(tree["TrackId"][tree_index_of_trackid[is_resolved]] == trackid[is_resolved])

        # Check that all halos on the same branch have the same TrackId
        i1 = 0
        while i1 < len(tree["GalaxyId"]):
            i2 = tree["EndMainBranchId"][i1] + 1
            assert np.all(tree["TrackId"][i1:i2] == tree["TrackId"][i1])
            i1 = i2
