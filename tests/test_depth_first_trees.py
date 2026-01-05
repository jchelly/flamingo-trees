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
