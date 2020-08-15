QUANTIZATION_SRC = $(wildcard plugin/quantization/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(QUANTIZATION_SRC))

QUANTIZATION_CUSRC = $(wildcard plugin/quantization/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(QUANTIZATION_CUSRC))