CXX=g++
CPPFLAGS=-fPIC -g -Wall -O1
LDLIBS=-O1 -larmadillo
LDFLAGS=-L$(DIR) -Wl,-rpath=$(DIR)
RM=rm -f
DIR=$(CURDIR)

SRCS=Tensor.cpp Index.cpp ternaryMera.cpp iDMRG.cpp
OBJS=$(subst .cpp,.o,$(SRCS))
LIBS=libtmera.so libidmrg.so
EXE=ex

all: $(EXE)

$(EXE): $(LIBS)
	$(CXX) -Wall -o $(EXE) $(LDFLAGS) $(LDLIBS) example.cpp -ltmera -lidmrg

libtmera.so: Tensor.o Index.o ternaryMera.o
	$(CXX) -shared -o libtmera.so Tensor.o Index.o ternaryMera.o $(LDLIBS)

libidmrg.so: Tensor.o Index.o iDMRG.o
	$(CXX) -shared -o libidmrg.so Tensor.o Index.o iDMRG.o $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend

clean:
	$(RM) $(OBJS)

dist-clean: clean
	$(RM) $(OBJS)

include .depend
