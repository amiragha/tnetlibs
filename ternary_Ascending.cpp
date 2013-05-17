#include "ternaryMera.h"

using namespace std;

Tensor ternary_Ascending (Tensor & Hamiltonian,
                          Tensor & Unitary, Tensor & Isometry){

    // extracting index information about input Tensors
    int in_card = Isometry.indeces[0].card;
    int out_card = Isometry.indeces[3].card;

    // we need UnitaryStar and IsometryStart as well
    Tensor UnitaryStar = Unitary;
    UnitaryStar.conjugate();
    Tensor IsometryStar = Isometry;
    IsometryStar.conjugate();
    // cout << "done" << endl;
    // cout <<endl << "printing Unitary" << endl;
    // Unitary.print();
    // cout << endl<<"printing Isometry"<<endl;
    // Isometry.print();
    // cout << "asdfasdf" << endl;
    // cout<<endl<<"printing unitary star" << endl;
    // Isometry.print(2);
    // cout << endl;
    // cout << endl<< "printing isometric star" <<endl;
    // cout << IsometryStar.values.size() << endl;
    // for (int i = 0; i < IsometryStar.values.size(); ++i)
    //     cout << IsometryStar.values[i] << endl;



    Tensor Temp; // for keeping the temporary results

    // needed inedeces
    Index o1("o1",out_card),o2("o2",out_card),
        o3("o3",out_card),o4("o4",out_card),
        u1("u1",in_card),u2("u2",in_card),
        u3("u3",in_card),u4("u4",in_card),
        t1("t1",in_card),t2("t2",in_card),
        t3("t3",in_card),t4("t4",in_card),
        t5("t5",in_card),t6("t6",in_card),
        t7("t7",in_card),t8("t8",in_card),t9("t1",in_card);

    vector<Index> h_idcs;
    vector<Index> u_idcs;
    vector<Index> us_idcs;
    vector<Index> t1_idcs;
    vector<Index> t1s_idcs;
    vector<Index> t2_idcs;
    vector<Index> t2s_idcs;

    // there are 3 different ascending Left Right Center

    //Left
    // vectors of indeces of Left
    h_idcs   = {t2, u1, t7, u3};
    u_idcs   = {u3, u2, t8, t9};
    us_idcs  = {u1, u2, t3, t4};
    t1_idcs  = {t1, t7, t8, o3};
    t1s_idcs = {t1, t2, t3, o1};
    t2_idcs  = {t9, t5, t6, o4};
    t2s_idcs = {t4, t5, t6, o2};
    // product-contraction
    IsometryStar.reIndex(t1s_idcs);
    Isometry.reIndex(t1_idcs);
    Hamiltonian.reIndex(h_idcs);
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);

    Temp = (IsometryStar * Isometry) * ((Hamiltonian * Unitary) * UnitaryStar);
    IsometryStar.reIndex(t2s_idcs);
    Isometry.reIndex(t2_idcs);

    Tensor Left = Temp * (IsometryStar * Isometry);

    //Center
    // vectors of indeces of Center
    h_idcs   = {u1, u2, u3, u4};
    u_idcs   = {u3, u4, t7, t8};
    us_idcs  = {u1, u2, t3, t4};
    t1_idcs  = {t1, t2, t7, o3};
    t1s_idcs = {t1, t2, t3, o1};
    t2_idcs  = {t8, t5, t6, o4};
    t2s_idcs = {t4, t5, t6, o2};
    // product-contraction
    IsometryStar.reIndex(t1s_idcs);
    Isometry.reIndex(t1_idcs);
    Hamiltonian.reIndex(h_idcs);
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);

    Temp = (IsometryStar * Isometry) * ((Hamiltonian * Unitary) * UnitaryStar);

    IsometryStar.reIndex(t2s_idcs);
    Isometry.reIndex(t2_idcs);

    Tensor Center = Temp * (IsometryStar * Isometry);

    //Right
    // vectors of indeces of Right
    h_idcs   = {u2, t5, u3, t9};
    u_idcs   = {u1, u3, t7, t8};
    us_idcs  = {u1, u2, t3, t4};
    t1_idcs  = {t1, t2, t7, o3};
    t1s_idcs = {t1, t2, t3, o1};
    t2_idcs  = {t8, t9, t6, o4};
    t2s_idcs = {t4, t5, t6, o2};
    // product-contraction
    IsometryStar.reIndex(t1s_idcs);
    Isometry.reIndex(t1_idcs);
    Hamiltonian.reIndex(h_idcs);
    Unitary.reIndex(u_idcs);
    UnitaryStar.reIndex(us_idcs);

    Temp = (IsometryStar * Isometry) * ((Hamiltonian * Unitary) * UnitaryStar);

    IsometryStar.reIndex(t2s_idcs);
    Isometry.reIndex(t2_idcs);

    Tensor Right = Temp * (IsometryStar * Isometry);

    vector<Index> output_Idxs =  {o1, o2, o3, o4};
    // for (int i = 0; i < Left.indeces.size(); ++i)
    //     cout << Left.indeces[i].name << "\t";
    // cout << endl;
    // for (int i = 0; i < Center.indeces.size(); ++i)
    //     cout << Center.indeces[i].name << "\t";
    // cout << endl;
    // for (int i = 0; i < Right.indeces.size(); ++i)
    //     cout << Right.indeces[i].name << "\t";
    // cout << endl;

    Temp = ((Right + Center) + Left)/3;
    Temp.rearrange(output_Idxs);
    // for (int i = 0; i < Temp.indeces.size(); ++i)
    //     cout << Temp.indeces[i].name << "\t";
    // cout << endl;

    return Temp;

}

//  LocalWords:  IsometryStart UnitaryStar
