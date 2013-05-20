#include "ternaryMera.h"

using namespace std;

double energy(Tensor & Hamiltonian, Tensor & DensityMatrix){
    // reading the initial indeces
    int card = Hamiltonian.indeces[0].card;
    vector<Index> hIdx = Hamiltonian.indeces;
    vector<Index> dIdx = Hamiltonian.indeces;

    // reIndexing for the correct contraction
    Index e1("e1",card),e2("e2",card),e3("e3",card),e4("e4",card);
    vector<Index> enIdx = {e1,e2,e3,e4};
    vector<Index> enIdx2 = {e3,e4,e1,e2};
    Hamiltonian.reIndex(enIdx);
    DensityMatrix.reIndex(enIdx2);
    Tensor energy = Hamiltonian * DensityMatrix;
    energy.print(1);

    // changing the indeces back to initial indeces
    Hamiltonian.reIndex(hIdx);
    DensityMatrix.reIndex(dIdx);
    return energy.values[0].real();
}
