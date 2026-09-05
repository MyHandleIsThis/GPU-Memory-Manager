# Makefile for CUDA Memory Manager
NVCC = nvcc
# C++17 is strictly required for std::optional
NVCCFLAGS = -std=c++17 -O3 -arch=sm_70
INCLUDES = -I.
# -lcuda is required for cuMemCreate, cuMemMap, etc.
LDFLAGS = -lcuda

TARGET = memory_manager
SRC = main.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)