#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/index_soap.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

activate flamingo_trees

# Simulation to do (based on job name)
sim="${SLURM_JOB_NAME}"
first_snap=0
last_snap=77

# Input SOAP catalogue
input_soap="/cosma8/data/dp004/flamingo/Runs/${sim}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"

# Output location
output_dir="/cosma8/data/dp004/jch/FLAMINGO/HBT-trees/${sim}"

# Set striping on output location
\mkdir -p "${output_dir}"
lfs setstripe --stripe-count=1 --stripe-size=8M "${outdir}"

# Run the code
mpirun -- python3 -m mpi4py -m flamingo_trees.build.index_soap \
       "${input_soap}" "${first_snap}" "${last_snap}" "${output_dir}/merger_tree.hdf5"
