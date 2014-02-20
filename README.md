TnetLibs
====
## Usage and Code structure

Currently there are two libraries for performing the calculation of
ternary MERA (according to
[arXiv:0707.1454](http://arxiv.org/abs/0707.1454)) and iDMRG
(according to [arXiv:0804.2509](http://arxiv.org/abs/0804.2509)). For
a simple example for both of them, take a look at `example.cpp`.

## compiling

for compiling just run:

```bash
$ make
```

This will make a `ex` *executable* file and two shared libraries,
`libtmera.so` and `libidmrg.so`.

### dependencies

1. [armadillo](http://arma.sourceforge.net/) must be in LIBRARY_PATH. Note that the libraries only compile with armadillo 4 up.
