#!/bin/env python

import h5py
import numpy as np
import pytest

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from flamingo_trees.build.index_soap import make_soap_index


@pytest.mark.mpi
def test_index_soap(tmp_path):
    """
    Run the SOAP indexing code and check the output
    """
    # Parameters for test run
    soap_format = "./tests/data/halo_properties_{snap_nr:04d}.hdf5"
    first_snap = 0
    last_snap = 8

    # Different MPI ranks generate different tmp paths!
    # Here everyone will use the one from rank zero.
    tmp_path = comm.bcast(tmp_path)

    # Run the code
    output_file = tmp_path / "merger_tree.hdf5"
    with h5py.File(output_file, "w", driver="mpio", comm=comm) as f:
        make_soap_index(soap_format, first_snap, last_snap, f)

    # Check the output. Do this on one rank only, for simplicity.
    ok = True
    if comm_rank == 0:
        with h5py.File(output_file, "r") as tree_file:
            for snap_nr in range(first_snap, last_snap+1):

                # Read TrackId from the SOAP file
                with h5py.File(soap_format.format(snap_nr=snap_nr), "r") as soap_file:
                    trackid = soap_file["InputHalos/HBTplus/TrackId"][...]

                # Read SOAPIndexOfTrackId from the merger tree file
                soap_index = tree_file[f"Snapshots/{snap_nr:04d}/SOAPIndexOfTrackId"][...]

                # SOAPIndexOfTrackId gives us the index where each TrackId is stored.
                nr_halos = len(trackid)
                ok = ok & np.all(soap_index[trackid] == np.arange(nr_halos, dtype=int))

    # Ensure that if anything went wrong, all ranks abort
    ok = comm.bcast(ok)
    assert ok

