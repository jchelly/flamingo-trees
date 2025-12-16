#!/bin/env python

import h5py
import pytest
from mpi4py import MPI
comm = MPI.COMM_WORLD

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
