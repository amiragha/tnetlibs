CXX=g++
CPPFLAGS=-g -std=c++11
LDLIBS=-larmadillo
LDFLAGS=
RM=rm -f

SRCS=test.cpp Tensor.cpp Index.cpp ternary_Ascending.cpp ternary_Descending.cpp \
	ternaryMera.cpp giveRandomDensity.cpp giveRandomTensors.cpp energy.cpp \
	ternary_Environment_Unit.cpp ternary_Environment_Iso.cpp
OBJS=$(subst .cpp,.o,$(SRCS))
EXE=test

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX)  $(OBJS) -o $(EXE) $(LDFLAGS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend

clean:
	$(RM) $(OBJS)

dist-clean: clean
	$(RM) $(OBJS)

include .depend
