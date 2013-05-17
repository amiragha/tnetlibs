#include "ternaryMera.h"

using namespace std;

Tensor ternary_Environment_Unit (Tensor & Hamiltonian,
                                 Tensor & DensityMatrix,
                                 Tensor & Unitary,
                                 Tensor & Isometry){
    // extracting index information about input Tensors
    int in_card = Isometry.indeces[0].card;
    int out_card = Isometry.indeces[3].card;

    // check for cardinality mismatch TO-DO

    // we need UnitaryStar and IsometryStart as well
    Tensor UnitaryStar = Unitary;
    UnitaryStar.conjugate();
    Tensor IsometryStar = Isometry;
    IsometryStar.conjugate();

    Tensor Temp; // for keeping the temporary results

    // needed inedeces
    Index o1("o1",in_card),o2("o2",in_card),
        o3("o3",in_card),o4("o4",in_card),
        u1("u1",in_card),u2("u2",in_card),
        u3("u3",in_card),u4("u4",in_card),
        t1("t1",in_card),t2("t2",in_card),
        t3("t3",in_card),t4("t4",in_card),t5("t5",in_card),
        d1("d1",out_card),d2("d2",out_card),
        d3("d3",out_card),d4("d4",out_card);

    vector<Index> output_Idxs =  {o1, o2, o3, o4};
    vector<Index> h_idcs;
    vector<Index> d_idcs;
    vector<Index> us_idcs;
    vector<Index> t1_idcs;
    vector<Index> t1s_idcs;
    vector<Index> t2_idcs;
    vector<Index> t2s_idcs;

    // Left
    // vector of indeces
    h_idcs   = {t2, u3, t5, o1};
    d_idcs   = {d3, d4, d1, d2};
    us_idcs  = {u3, o2, u1, u2};
    t1_idcs  = {t1, t5, o3, d3};
    t1s_idcs = {t1, t2, u1, d1};
    t2_idcs  = {o4, t3, t4, d4};
    t2s_idcs = {u2, t3, t4, d2};
    // product-contraction
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    DensityMatrix.reIndex(d_idcs);
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);

    Temp = (Isometry * IsometryStar) * DensityMatrix;

    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);

    Tensor Left = (((Isometry * IsometryStar) * Temp) * UnitaryStar) * Hamiltonian;
    Left.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Unitary.reIndex(output_Idxs);
    // (Unitary * Left).print(1);


    // Center
    // vector of indeces
    h_idcs   = {u3, u4, o1, o2};
    d_idcs   = {d3, d4, d1, d2};
    us_idcs  = {u3, u4, u1, u2};
    t1_idcs  = {t1, t2, o3, d3};
    t1s_idcs = {t1, t2, u1, d1};
    t2_idcs  = {o4, t3, t4, d4};
    t2s_idcs = {u2, t3, t4, d2};
    // product-contraction
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    DensityMatrix.reIndex(d_idcs);
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);

    Temp = (Isometry * IsometryStar) * DensityMatrix;

    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);

    Tensor Center = (Isometry * IsometryStar) * Temp * UnitaryStar * Hamiltonian;
    Center.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Unitary.reIndex(output_Idxs);
    // (Unitary * Center).print(1);

    // Right
    // vector of indeces
    h_idcs   = {u3, t3, o2, t5};
    d_idcs   = {d3, d4, d1, d2};
    us_idcs  = {o1, u3, u1, u2};
    t1_idcs  = {t1, t2, o3, d3};
    t1s_idcs = {t1, t2, u1, d1};
    t2_idcs  = {o4, t5, t4, d4};
    t2s_idcs = {u2, t3, t4, d2};
    // product-contraction
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    DensityMatrix.reIndex(d_idcs);
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);

    Temp = (Isometry * IsometryStar) * DensityMatrix;

    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);

    Tensor Right = (((Isometry * IsometryStar) * Temp) * UnitaryStar) * Hamiltonian;
    Right.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Unitary.reIndex(output_Idxs);
    // (Unitary * Right).print(1);

    return (Left + Center + Right)/3;
}
