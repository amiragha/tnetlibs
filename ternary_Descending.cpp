#include "ternaryMera.h"

using namespace std;
using namespace arma;

Tensor ternary_Descending (Tensor & DensityMatrix,
                           Tensor & Unitary, Tensor & Isometry){

    // extracting index information about input Tensors
    int in_card = Isometry.indeces[3].card;
    int out_card = Isometry.indeces[0].card;
    vector<Index> d_input = DensityMatrix.indeces;
    vector<Index> u_input = Unitary.indeces;
    vector<Index> t_input = Isometry.indeces;

    // we need UnitaryStar and IsometryStart as well
    Tensor UnitaryStar = Unitary;
    UnitaryStar.conjugate();
    Tensor IsometryStar = Isometry;
    IsometryStar.conjugate();

    Tensor Temp; // for keeping the temporary results

    // needed indeces
    Index o1("o1",out_card),o2("o2",out_card),
        o3("o3",out_card),o4("o4",out_card),
        u1("u1",out_card),u2("u2",out_card),
        u3("u3",out_card),u4("u4",out_card),u5("u5",out_card),
        t1("t1",out_card),t2("t2",out_card),
        t3("t3",out_card),t4("t4",out_card),
        i1("i1",in_card),i2("i2",in_card),
        i3("i3",in_card),i4("i4",in_card);

    vector<Index> output_Idxs = {o1, o2, o3, o4};

    vector<Index> d_idcs;
    vector<Index> u_idcs;
    vector<Index> us_idcs;
    vector<Index> t1_idcs;
    vector<Index> t1s_idcs;
    vector<Index> t2_idcs;
    vector<Index> t2s_idcs;


    // there are 3 different ascending Left Right Center

    //Left
    // vectors of indeces of Left
    d_idcs   = {i1, i2, i3, i4};
    u_idcs   = {o2, u1, u2, u3};
    us_idcs  = {o4, u1, u4, u5};
    t1_idcs  = {t1, o1, u2, i1};
    t1s_idcs = {t1, o3, u4, i3};
    t2_idcs  = {u3, t3, t2, i2};
    t2s_idcs = {u5, t3, t2, i4};
    // product-contraction
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);
    DensityMatrix.reIndex(d_idcs);
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);

    Temp = (Isometry * IsometryStar) * DensityMatrix;

    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);

    Tensor Left = (Isometry * IsometryStar) * Temp * (Unitary * UnitaryStar);
    Left.rearrange(output_Idxs);
    //Center
    // vectors of indeces of Center
    d_idcs   = {i1, i2, i3, i4};
    u_idcs   = {o1, o2, u1, u2};
    us_idcs  = {o3, o4, u3, u4};
    t1_idcs  = {t2, t1, u1, i1};
    t1s_idcs = {t2, t1, u3, i3};
    t2_idcs  = {u2, t4, t3, i2};
    t2s_idcs = {u4, t4, t3, i4};
    // product-contraction
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);
    DensityMatrix.reIndex(d_idcs);
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);

    Temp = (Isometry * IsometryStar) * DensityMatrix;

    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);

    Tensor Center = Unitary * ((Isometry * IsometryStar) * Temp) * UnitaryStar;

    //Right
    // vectors of indeces of Right
    d_idcs   = {i1, i2, i3, i4};
    u_idcs   = {u1, o1, u2, u3};
    us_idcs  = {u1, o3, u4, u5};
    t1_idcs  = {t2, t1, u2, i1};
    t1s_idcs = {t2, t1, u4, i3};
    t2_idcs  = {u3, o2, t3, i2};
    t2s_idcs = {u5, o4, t3, i4};
    // product-contraction
    Isometry.reIndex(t2_idcs);
    IsometryStar.reIndex(t2s_idcs);
    DensityMatrix.reIndex(d_idcs);
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);

    Temp = (Isometry * IsometryStar) * DensityMatrix;

    Isometry.reIndex(t1_idcs);
    IsometryStar.reIndex(t1s_idcs);

    Tensor Right = (Isometry * IsometryStar) * Temp * (Unitary * UnitaryStar);
    Right.rearrange(output_Idxs);

    // for (int i = 0; i < Left.indeces.size(); ++i)
    //     cout << Left.indeces[i].name << "\t";
    // cout << endl;
    // for (int i = 0; i < Center.indeces.size(); ++i)
    //     cout << Center.indeces[i].name << "\t";
    // cout << endl;
    // for (int i = 0; i < Right.indeces.size(); ++i)
    //     cout << Right.indeces[i].name << "\t";
    // cout << endl;
    // cout << "calculation of Left, Center and Right done" << endl;
    Temp = ((Right + Center) + Left)/3;

    // reIndexing to original input
    DensityMatrix.reIndex(d_input);
    Unitary.reIndex(u_input);
    Isometry.reIndex(t_input);

    return Temp;

}
