GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"

CPPSRC = 
CUSRC =
BASEEXE = main_COAL
VARIANT ?= DEFAULT
# CUDA source files (compiled with cudacc)
EXECUTABLE = ./$(BASEEXE)
CUSRC += *.cu 
OPTS = -O3
PTX_GEN =  $(TRANSFORM_SCRIPT)/util/ptx_transform_coal

CUOPTS = $(GENCODE_SM70)
INC = -I../../include/mem_alloc
LIBS = 

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: $(OBJS)
	# Generate dryrun file
	nvcc --dryrun --keep $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC)  $(INC) -o $(EXECUTABLE) $(LIBS) 2> dryrun.sh
	# Remove all lines before/including cicc
	sed -i '1,/cicc/d' dryrun.sh
	sed -i '/cicc/d' dryrun.sh
	# Remove rm line
	sed -i '/rm/d' dryrun.sh
	# Remove leading comment
	cut -c 3- dryrun.sh > dryrun1.sh
	mv dryrun1.sh dryrun.sh
	nvcc  --keep   $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC)  $(INC) -o $(EXECUTABLE) $(LIBS) 
	$(PTX_GEN)/generator.py main.ptx
	cp main.ptx_coal main.ptx
	sh dryrun.sh
	rm -f *cpp* *fatbin* *cudafe*  *cubin* *.o *.module_id *dlink*

$(OBJS):
	g++ -c $(CPPSRC) $(OPTS)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleancucpp:
	rm -f *.cu.cpp
