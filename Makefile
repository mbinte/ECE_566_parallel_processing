CC ?= cc
MPICC ?= mpicc
NVCC ?= nvcc
OMP_CC ?= $(CC)
OMP_FLAGS ?= -fopenmp
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
CPPFLAGS += -D_POSIX_C_SOURCE=200809L
NVFLAGS ?= -O3
BIN := build
GAUSS := hw2-shared-memory/gauss_shared.c
CONV := project-convolution/convolve.c
FFT := project-convolution/fft.c

.PHONY: all cpu mpi cuda clean
all: cpu mpi
cpu: $(addprefix $(BIN)/,gauss_serial gauss_pthreads gauss_openmp conv_serial)
mpi: $(addprefix $(BIN)/,gauss_mpi conv_sendrecv conv_collective conv_tasks)
cuda: $(addprefix $(BIN)/,hello vector_add matrix_sums matrix_norm)

$(BIN):
	mkdir -p $@

$(BIN)/gauss_serial: $(GAUSS) common/gauss.h | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ -lm
$(BIN)/gauss_pthreads: $(GAUSS) common/gauss.h | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DUSE_PTHREAD -pthread $< -o $@ -lm
$(BIN)/gauss_openmp: $(GAUSS) common/gauss.h | $(BIN)
	$(OMP_CC) $(CPPFLAGS) $(CFLAGS) $(OMP_FLAGS) $< -o $@ -lm
$(BIN)/gauss_mpi: hw3-mpi/gauss_mpi.c common/gauss.h | $(BIN)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $< -o $@ -lm
$(BIN)/conv_serial: $(CONV) $(FFT) | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ -lm
$(BIN)/conv_sendrecv: $(CONV) $(FFT) | $(BIN)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -DPOINT_TO_POINT $< -o $@ -lm
$(BIN)/conv_collective: $(CONV) $(FFT) | $(BIN)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -DCOLLECTIVE $< -o $@ -lm
$(BIN)/conv_tasks: $(CONV) $(FFT) | $(BIN)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -DTASKS $< -o $@ -lm

$(BIN)/%: hw4-cuda/%.cu common/cuda_check.cuh | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@
$(BIN)/matrix_norm: extra-cuda/matrix_norm.cu common/cuda_check.cuh | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	rm -rf $(BIN)
