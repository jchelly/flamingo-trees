#!/bin/env python

import h5py
import pytest

from flamingo_trees.build.index_soap import make_soap_index


def try_index_soap(tmp_path):
    """
    Run the SOAP indexing code and check the output
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Parameters for test run
    soap_format = "./tests/data/halo_properties_{snap_nr:04d}.hdf5"
    first_snap = 0
    last_snap = 8

    # Run the code
    output_file = tmp_path / "merger_tree.hdf5"
    with h5py.File(output_file, "w", driver="mpio", comm=comm) as f:
        make_soap_index(soap_format, first_snap, last_snap, f)


@pytest.mark.mpi
def test_index_soap_np1(tmp_path):
    try_index_soap(tmp_path)
