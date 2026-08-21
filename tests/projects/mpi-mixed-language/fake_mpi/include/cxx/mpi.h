#ifndef BLT_TEST_FAKE_MPI_CXX_MPI_H
#define BLT_TEST_FAKE_MPI_CXX_MPI_H

#define BLT_FAKE_MPI_CXX_HEADER 1

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

#define MPI_COMM_WORLD 0
#define MPI_INT 0
#define MPI_SUM 0

static inline int MPI_Init(int *, char ***)
{
    return 0;
}

static inline int MPI_Comm_rank(MPI_Comm, int *rank)
{
    *rank = 0;
    return 0;
}

static inline int MPI_Comm_size(MPI_Comm, int *size)
{
    *size = 4;
    return 0;
}

static inline int MPI_Reduce(const void *sendbuf,
                             void *recvbuf,
                             int,
                             MPI_Datatype,
                             MPI_Op,
                             int,
                             MPI_Comm)
{
    *static_cast<int *>(recvbuf) = *static_cast<const int *>(sendbuf) * 4;
    return 0;
}

static inline int MPI_Finalize()
{
    return 0;
}

#endif
