#include "ternaryMera.h"

using namespace std;

Tensor ternary_Environment_Iso (Tensor & Hamiltonian,
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

    // cout << "checking Isometry and Unitary" << endl;
    // (Isometry * IsometryStar).print(1);
    // (Unitary * UnitaryStar).print(1);
    Tensor Temp; // for keeping the temporary results

    // needed inedeces
    Index o1("o1",in_card),o2("o2",in_card),
        o3("o3",in_card),o4("o4",out_card),
        u1("u1",in_card),u2("u2",in_card),
        u3("u3",in_card),u4("u4",in_card),
        u5("u5",in_card),u6("u6",in_card),u7("u7",in_card),
        t1("t1",in_card),t2("t2",in_card),t3("t3",in_card),
        d1("d1",out_card),d2("d2",out_card),d3("d3",out_card);

    vector<Index> output_Idxs =  {o1, o2, o3, o4};
    vector<Index> h_idcs;
    vector<Index> d_idcs;
    vector<Index> u_idcs;
    vector<Index> us_idcs;
    vector<Index> t1_idcs;
    vector<Index> t1s_idcs;
    vector<Index> t2_idcs;
    vector<Index> t2s_idcs;

    // there are 6 different Environment Isometry calculations
    // Left_T1 Center_T1 Right_T1
    // Left_T2 Center_T2 Right_T2

    // Left_T1
    // vectors of indeces
    h_idcs   = {t1, u4, o2, u6};
    d_idcs   = {o4, d3, d1, d2};
    u_idcs   = {u6, u5, o3, u3};
    us_idcs  = {u4, u5, u1, u2};
    t1s_idcs = {o1, t1, u1, d1};
    t2_idcs  = {u3, t2, t3, d3};
    t2s_idcs = {u2, t2, t3, d2};
    // product-contraction
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);
    DensityMatrix.reIndex(d_idcs);

    //Temp = ((Unitary * Hamiltonian * UnitaryStar) *
    //        (Isometry * IsometryStar) * DensityMatrix);
    Temp = Unitary * Hamiltonian * UnitaryStar;
    // for (int i = 0; i < Temp.indeces.size(); ++i)
    //     cout << Temp.indeces[i].name << "\t";
    // std::cout << std::endl;
    Temp = Temp * (Isometry * IsometryStar) * DensityMatrix;
    // Temp.print(in_card*in_card);
    // for (int i = 0; i < Temp.indeces.size(); ++i)
    //     cout << Temp.indeces[i].name << "\t";
    // std::cout << std::endl;
    // Temp.print(out_card*out_card);

    IsometryStar.reIndex(t1s_idcs);

    Tensor Left_T1 = IsometryStar * Temp;
    Left_T1.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Isometry.reIndex(output_Idxs);
    // (Left_T1 * Isometry).print(1);

    // Center_T1
    // vectors of indeces
    h_idcs   = {u4, u5, u6, u7};
    d_idcs   = {o4, d3, d1, d2};
    u_idcs   = {u6, u7, o3, u3};
    us_idcs  = {u4, u5, u1, u2};
    t1s_idcs = {o1, o2, u1, d1};
    t2_idcs  = {u3, t1, t2, d3};
    t2s_idcs = {u2, t1, t2, d2};
    // product-contraction
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);
    DensityMatrix.reIndex(d_idcs);

    Temp = ((Unitary * Hamiltonian * UnitaryStar) *
            (Isometry * IsometryStar) * DensityMatrix);

    IsometryStar.reIndex(t1s_idcs);

    Tensor Center_T1 = IsometryStar * Temp;
    Center_T1.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Isometry.reIndex(output_Idxs);
    // (Center_T1 * Isometry).print(1);

    // Right_T1
    // vectors of indeces
    h_idcs   = {u5, t1, u6, t3};
    d_idcs   = {o4, d3, d1, d2};
    u_idcs   = {u4, u6, o3, u3};
    us_idcs  = {u4, u5, u1, u2};
    t1s_idcs = {o1, o2, u1, d1};
    t2_idcs  = {u3, t3, t2, d3};
    t2s_idcs = {u2, t1, t2, d2};
    // product-contractio
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);
    DensityMatrix.reIndex(d_idcs);

    Temp = ((Unitary * Hamiltonian * UnitaryStar) *
            (Isometry * IsometryStar) * DensityMatrix);

    IsometryStar.reIndex(t1s_idcs);

    Tensor Right_T1 = IsometryStar * Temp;
    Right_T1.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Isometry.reIndex(output_Idxs);
    // (Right_T1 * Isometry).print(1);

    // Left_T2
    // vectors of indeces
    h_idcs   = {t2, u4, t3, u6};
    d_idcs   = {d3, o4, d1, d2};
    u_idcs   = {u6, u5, u3, o1};
    us_idcs  = {u4, u5, u1, u2};
    t1_idcs  = {t1, t3, u3, d3};
    t1s_idcs = {t1, t2, u1, d1};
    t2s_idcs = {u2, o2, o3, d2};
    // product-contraction
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);
    DensityMatrix.reIndex(d_idcs);

    Temp = ((Isometry * IsometryStar) *
            (Unitary * Hamiltonian * UnitaryStar) * DensityMatrix);

    IsometryStar.reIndex(t2s_idcs);

    Tensor Left_T2 = Temp * IsometryStar;
    Left_T2.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Isometry.reIndex(output_Idxs);
    // (Left_T2 * Isometry).print(1);

    // Center_T2
    // vectors of indeces
    h_idcs   = {u4, u5, u6, u7};
    d_idcs   = {d3, o4, d1, d2};
    u_idcs   = {u6, u7, u3, o1};
    us_idcs  = {u4, u5, u1, u2};
    t1_idcs  = {t1, t2, u3, d3};
    t1s_idcs = {t1, t2, u1, d1};
    t2s_idcs = {u2, o2, o3, d2};
    // product-contraction
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);
    DensityMatrix.reIndex(d_idcs);

    Temp = ((Isometry * IsometryStar) *
            (Unitary * Hamiltonian * UnitaryStar) * DensityMatrix);

    IsometryStar.reIndex(t2s_idcs);

    Tensor Center_T2 = Temp * IsometryStar;
    Center_T2.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Isometry.reIndex(output_Idxs);
    // (Center_T2 * Isometry).print(1);
    // Right_T2
    // vectors of indeces
    h_idcs   = {u5, t3, u6, o2};
    d_idcs   = {d3, o4, d1, d2};
    u_idcs   = {u4, u6, u3, o1};
    us_idcs  = {u4, u5, u1, u2};
    t1_idcs  = {t1, t2, u3, d3};
    t1s_idcs = {t1, t2, u1, d1};
    t2s_idcs = {u2, t3, o3, d2};
    // product-contraction
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);
    Hamiltonian.reIndex(h_idcs);
    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);
    DensityMatrix.reIndex(d_idcs);

    Temp = ((Isometry * IsometryStar) *
            (Unitary * Hamiltonian * UnitaryStar) * DensityMatrix);

    IsometryStar.reIndex(t2s_idcs);

    Tensor Right_T2 = Temp * IsometryStar;
    Right_T2.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // Isometry.reIndex(output_Idxs);
    // (Isometry * Right_T2).print(1);
    // for (int i = 0; i < Left_T1.indeces.size(); ++i)
    //     cout << Left_T1.indeces[i].name << "\t";
    // cout << endl;
    return (Left_T2 + Center_T2 + Right_T2 + Left_T1 + Center_T1 + Right_T1)/6;
}
