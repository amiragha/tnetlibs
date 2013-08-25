#include <iostream>
#include "Tensor.h"
#include "ternaryMera.h"
#include "iDMRG.h"

using namespace std;
using namespace arma;

int main(int argc, char *argv[])
{
    // defining some indeces for test and run
    // Index a("a", 3), b("b", 2), c("c", 2), d("d",2), e("e",3),f("f",2);

    // // defining pauli matrices
    // cx_mat PauliX,PauliY,PauliZ,I;
    // PauliX << cx_d(0.0,0.0) << cx_d(1.0,0.0) << endr
    //        << cx_d(1.0,0.0) << cx_d(0.0,0.0) << endr;

    // PauliY << cx_d(0.0,0.0) << cx_d(0.0,-1.0) << endr
    //        << cx_d(0.0,1.0) << cx_d(0.0,0.0) << endr;

    // PauliZ << cx_d(1.0,0.0) << cx_d(0.0,0.0) << endr
    //        << cx_d(0.0,0.0) << cx_d(-1.0,0.0) << endr;

    // cx_mat I2(2,2);
    // I2.eye();

    // cx_mat I4(4,4);
    // I4.eye();

    // // making the ITF Hamiltonian
    // cx_mat ITF = -kron(PauliZ,PauliZ)-kron(PauliX,I2)/2-kron(I2,PauliX)/2;

    // cout << ITF<<endl;

    // TernaryMera test(ITF, 2, 4, true);
    // test.buOptimize(30,true, true);

    IDMRG testidmrg(20);
    return 0;

}
