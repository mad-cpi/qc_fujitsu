#!/usr/bin/env bash

# Usage:
#     mpirun -n <num_ranks> -npernode 1 job.sh <path/to/venv> <command> <command arguments>
#
# Example:
#     (python): mpirun -n 2 -npernode 1 job.sh ~/example/venv python ./sample.py
#     (pytest): mpirun -n 2 -npernode 1 job.sh ~/example/venv pytest

# Setting the Environment Variables required to use MPI in the quantum simulator system
export UCX_IB_MLX5_DEVX=no
export OMP_PROC_BIND=TRUE

# Number of threads used for OpenMP parallelization
# Due to the OpenMP multi-threading behavior, some libraries (such as scipy) can introduce 
# small calculation errors, which can lead to inconsistent calculation results between processes 
# during MPI execution. When using mpiQulacs, set it to 1.
export OMP_NUM_THREADS=1

# Number of threads used by mpiQulacs (if not set, the value of OMP_NUM_THREADS is used)
export QULACS_NUM_THREADS=48

#
source $1/bin/activate
shift

### Workaround for the glibc bug (https://bugzilla.redhat.com/show_bug.cgi?id=1722181)).
if [ -z "${LD_PRELOAD}" ]; then
    export LD_PRELOAD=/lib64/libgomp.so.1
else
    export LD_PRELOAD=/lib64/libgomp.so.1:$LD_PRELOAD
fi

#
LSIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}
LRANK=${OMPI_COMM_WORLD_LOCAL_RANK}
COM=$1
shift

if [ $LSIZE -eq 1 ]; then
    numactl -m 0-3 -N 0-3 ${COM} "$@"
elif [ $LSIZE -eq 4 ]; then
    numactl -N ${LRANK} -m ${LRANK} ${COM} "$@"
else
    ${COM} "$@"
fi