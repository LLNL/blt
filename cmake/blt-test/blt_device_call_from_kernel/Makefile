NVCC_FLAGS = -m64 -g -G -O0 -arch sm_60 -std c++11 --expt-extended-lambda -Xcompiler -g -lineinfo -I$(CUDA_PATH)/include
NVCC_COMP_FLAGS = $(NVCC_FLAGS) -x cu -dc
NVCC_LINK_FLAGS = $(NVCC_FLAGS) -dlink

ifeq ($(CUDA_PATH),)
  CUDA_PATH=/usr/local/cuda
endif

bad:
	$(CUDA_PATH)/bin/nvcc $(NVCC_COMP_FLAGS) Child.cxx -o Child.o
	$(CUDA_PATH)/bin/nvcc $(NVCC_COMP_FLAGS) Parent.cxx -o Parent.o
	$(CUDA_PATH)/bin/nvcc $(NVCC_LINK_FLAGS) Child.o Parent.o -o testprob_static_intermediate_link.o
	ar qc libtestprob.a Child.o Parent.o testprob_static_intermediate_link.o
	ranlib libtestprob.a
	$(CUDA_PATH)/bin/nvcc $(NVCC_COMP_FLAGS) CudaTests.cxx -o testExec.o
	$(CUDA_PATH)/bin/nvcc $(NVCC_LINK_FLAGS) testExec.o -o testExec_intermediate_link.o
	g++ -g testExec.o testExec_intermediate_link.o -o testExec $(CUDA_PATH)/lib64/libcudart_static.a -ldl libtestprob.a -lcuda -lpthread -lrt $(CUDA_PATH)/lib64/libcudart_static.a -ldl $(CUDA_PATH)/lib64/libcudadevrt.a

good:
	$(CUDA_PATH)/bin/nvcc $(NVCC_COMP_FLAGS) Child.cxx -o Child.o
	$(CUDA_PATH)/bin/nvcc $(NVCC_COMP_FLAGS) Parent.cxx -o Parent.o
	$(CUDA_PATH)/bin/nvcc $(NVCC_FLAGS) --lib Child.o Parent.o -o libtestprob.a
	$(CUDA_PATH)/bin/nvcc $(NVCC_COMP_FLAGS) CudaTests.cxx -o testExec.o
	$(CUDA_PATH)/bin/nvcc $(NVCC_FLAGS) testExec.o libtestprob.a -o testExec

run:
	cuda-memcheck ./testExec

clean:
	rm -rf *.o *.a testExec

tar:
	tar -czvf test.tgz *
