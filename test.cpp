#include <iostream>
#include "Tensor.h"
#include "ternaryMera.h"

using namespace std;
using namespace arma;

int main(int argc, char *argv[])
{
    Index a("a", 3), b("b", 2), c("c", 2), d("d",2), e("e",3),f("f",2);
    // vector<Index> v = {a,b,c};

    // Tensor T(v);
    // T.values =
    //     {cx_d(111.0,0.0), cx_d(211.0,0.0), cx_d(311.0,0.0), cx_d(121.0,0.0),
    //      cx_d(221.0,0.0), cx_d(321.0,0.0), cx_d(112.0,0.0), cx_d(212.0,0.0),
    //      cx_d(312.0,0.0), cx_d(122.0,0.0), cx_d(222.0,0.0), cx_d(322.0,0.0)};
    // T.print(4);
    // vector<Index> row = {b};
    // vector<Index> col = {c, a};
    // T.toMat(row, col);
    // cout << endl <<T.matRepresentation << endl;



    // // testing the contraction
    // vector<Index> v2 = {a, c, d};
    // Tensor T2(v2);
    // T2.values =
    //     {cx_d(111.0,1.0), cx_d(211.0,1.0), cx_d(311.0,1.0), cx_d(121.0,1.0),
    //      cx_d(221.0,1.0), cx_d(321.0,1.0), cx_d(112.0,1.0), cx_d(212.0,1.0),
    //      cx_d(312.0,1.0), cx_d(122.0,1.0), cx_d(222.0,1.0), cx_d(322.0,1.0)};


    // Tensor result = T2*T;
    // for (int i = 0; i < result.values.size() ; ++i)
    //     cout << result.values[i].real() << "\t";
    // cout << endl;

    // cout << (result).matRepresentation << endl;

    // Tensor T2Star = T2;
    // T2Star.conjugate();
    // T2.print(4);
    // cout << endl;
    // T2Star.print(4);
    // cout << endl;
    // (T2*T2Star).print(4);
    // cout << endl;

    // // testing the fromMat function
    // cx_mat A = randu<cx_mat>(3,3);
    // cout << A;
    // vector<Index> v3 = {a, e};
    // Tensor T3(v3);
    // vector<Index> row2(1,a);
    // vector<Index> col2(1,e);
    // T3.fromMat(A, row2,col2);
    // T3.print(4);
    // cout << endl;

    // defining pauli matrices
    cx_mat PauliX,PauliY,PauliZ,I;
    PauliX << cx_d(0.0,0.0) << cx_d(1.0,0.0) << endr
           << cx_d(1.0,0.0) << cx_d(0.0,0.0) << endr;

    PauliY << cx_d(0.0,0.0) << cx_d(0.0,-1.0) << endr
           << cx_d(0.0,1.0) << cx_d(0.0,0.0) << endr;

    PauliZ << cx_d(1.0,0.0) << cx_d(0.0,0.0) << endr
           << cx_d(0.0,0.0) << cx_d(-1.0,0.0) << endr;

    I << cx_d(1.0,0.0) << cx_d(0.0,0.0) << endr
      << cx_d(0.0,0.0) << cx_d(1.0,0.0) << endr;

    cx_mat ITF = -kron(PauliZ,PauliZ)-kron(PauliX,I)/2-kron(I,PauliX)/2;
    vec eigenvals = eig_sym(ITF);
    cx_mat I4(4,4);
    I4.eye();
    ITF = ITF - max(eigenvals)*I4;
    cout << ITF<<endl;
    vector<Index> vh = {b,c,d,f};
    vector<Index> vhr = {b,c};
    vector<Index> vhc = {d,f};

    Tensor H(vh);
    //H.fromMat(ITF,vhr,vhc);
    H.values =
        {cx_d(1.0,0.0), cx_d(0.0,0.0), cx_d(0.0,0.0), cx_d(0.0,0.0),
         cx_d(0.0,0.0), cx_d(1.0,0.0), cx_d(0.0,0.0), cx_d(0.0,0.0),
         cx_d(0.0,0.0), cx_d(0.0,0.0), cx_d(1.0,0.0), cx_d(0.0,0.0),
         cx_d(0.0,0.0), cx_d(0.0,0.0), cx_d(0.0,0.0), cx_d(1.0,0.0)};

    ternaryMera(H,9);

    return 0;

}
