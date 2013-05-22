CXX=g++
CPPFLAGS=-g
LDLIBS=-larmadillo
LDFLAGS=
RM=rm -f

SRCS=test.cpp Tensor.cpp Index.cpp ternaryMera.cpp
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
