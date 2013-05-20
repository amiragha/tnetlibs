#include "ternaryMera.h"

using namespace std;
using namespace arma;

void ternaryMera (Tensor & Hamiltonian0, int max_card){

    // note that max_card must be more than initial c*c*c
    // extracting Hamiltonian indeces
    vector<Index> h_input = Hamiltonian0.indeces;
    int c0,c1,c2,cn0,cn1,cn2;
    // level 0
    // defining the cardinality
    c0 = h_input[0].card;
    cn0 = c0*c0*c0;
    //generating random Unitary and Isometry Tensors
    Tensor Unitary0, Isometry0;
    giveRandomTensors(Unitary0,Isometry0,c0,cn0);

    // level 1
    // defining the cardinality
    c1 = cn0;
    cn1= max_card;
    //generating random Unitary and Isometry Tensors
    Tensor Unitary1, Isometry1;
    giveRandomTensors(Unitary1, Isometry1,c1,cn1);

    // level 2
    // defining the cardinality
    c2 = max_card;
    cn2 = max_card;
    //generating random Unitary and Isometry Tensors
    Tensor Unitary2,Isometry2;
    giveRandomTensors(Unitary2, Isometry2, c2, cn2);

    cout << "Unitary and Isometry Tensors initialization finished" << endl;

    // generating random Density Matrix
    Tensor DensityMatrix2;
    giveRandomDensity(DensityMatrix2,max_card);
    //DensityMatrix.print(cn2*cn2);
    // for (int i = 0; i < DensityMatrix3.indeces.size(); ++i)
    //     cout << DensityMatrix3.indeces[i].name << "\t";
    // cout <<endl;
    cout << "DensityMatrix initialization finished" << endl;


    cout << "preforming the descending" << endl;

    cout << "second level denity Matrix " << endl;

    for (int i = 0; i < 8; ++i)
        {
            // cout << endl;
            // cout << i+1 <<"\t";
            // for (int j = 0; j < 10; ++j)
            //     cout << DensityMatrix2.values[100*j] << "\t";
            DensityMatrix2 = ternary_Descending(DensityMatrix2, Unitary2, Isometry2);
        }
    cout << endl;

    cout << "first level density matrix" << endl;
    Tensor DensityMatrix1 = ternary_Descending(DensityMatrix2,Unitary1,Isometry1);

    cout << "zeroth level density matrix" << endl;
    Tensor DensityMatrix0 = ternary_Descending(DensityMatrix1,Unitary0,Isometry0);

    cout << "performing ascending" << endl;
    cout << "zeroth level Hamiltonian" << endl;
    //Hamiltonian0.print(c0*c0);
    Tensor Hamiltonian1 = ternary_Ascending(Hamiltonian0, Unitary0, Isometry0);
    cout << "first level hamiltonian" << endl;
    //Hamiltonian1.print(c1*c1);

    Tensor Hamiltonian2 = ternary_Ascending(Hamiltonian1, Unitary1, Isometry1);
    cout << "second level hamiltonian" << endl;
    //Hamiltonian2.print(c2*c2);

    Tensor Hamiltonian3 = ternary_Ascending(Hamiltonian2, Unitary2, Isometry2);
    cout << "third level hamiltonian" << endl;
    //Hamiltonian3.print(cn2*cn2);

    Tensor Hamiltonian4 = ternary_Ascending(Hamiltonian3, Unitary2, Isometry2);
    cout << "fourth level hamiltonian" << endl;
    //Hamiltonian4.print(cn2*cn2);

    cout << "energy calculations" << endl;

    cout << "energy at level 4 is :" << endl;
    energy(Hamiltonian4,DensityMatrix2);
    cout << "energy at level 3 is :" << endl;
    energy(Hamiltonian3,DensityMatrix2);
    cout << "energy at level 2 is :" << endl;
    energy(Hamiltonian2,DensityMatrix2);
    cout << "energy at level 1 is :" << endl;
    energy(Hamiltonian1,DensityMatrix1);
    //cout << "energy at level 0 is :" << endl;
    //energy(Hamiltonian0,DensityMatrix0);

    Hamiltonian0.printIndeces();
    Hamiltonian1.printIndeces();
    Hamiltonian2.printIndeces();
    Hamiltonian3.printIndeces();
    DensityMatrix0.printIndeces();
    DensityMatrix1.printIndeces();
    DensityMatrix2.printIndeces();

    Tensor Iso_env, Uni_env;
    cout << "performing the optimization" << endl;
    vector<Index> Ienvr;
    vector<Index> Ienvc;
    vector<Index> Uenvr;
    vector<Index> Uenvc;
    cx_mat U, V;
    vec s;
    int optN = 4;
    // iteration loop
    for (int iter = 0; iter < 10; ++iter)
        {
            cout << endl << "starting iteration number : " << iter+1 << " ==> "<< endl;
            cout <<         "------------------------------" << endl;
            cout << "Level 0 :" << endl;

            for (int number = 0; number < optN; ++number)
                {
                    Iso_env = ternary_Environment_Iso(Hamiltonian0,DensityMatrix1,Unitary0,Isometry0);
                    Ienvr = vector<Index> (Iso_env.indeces.begin(), Iso_env.indeces.begin()+3);
                    Ienvc = vector<Index> (Iso_env.indeces.begin()+3, Iso_env.indeces.begin()+4);
                    svd(U,s,V,Iso_env.toMat(Ienvr, Ienvc));
                    Isometry0.fromMat(-V*U.submat(span(),span(0,cn0-1)).t(),Ienvr, Ienvc);
                }
            Hamiltonian1 = ternary_Ascending(Hamiltonian0, Unitary0, Isometry0);
            cout << "energy at level 1 is :";
            energy(Hamiltonian1,DensityMatrix1);

            for (int number = 0; number < optN; ++number)
                {
                    Uni_env = ternary_Environment_Unit(Hamiltonian0,DensityMatrix1,Unitary0,Isometry0);
                    Uenvr = vector<Index> (Uni_env.indeces.begin(), Uni_env.indeces.begin()+2);
                    Uenvc = vector<Index> (Uni_env.indeces.begin()+2, Uni_env.indeces.begin()+4);
                    svd(U,s,V,Uni_env.toMat(Uenvr, Uenvc));
                    Unitary0.fromMat(-(V*U.t()).st(),Uenvr, Uenvc);
                }
            Hamiltonian1 = ternary_Ascending(Hamiltonian0, Unitary0, Isometry0);
            cout << "energy at level 1 is :";
            energy(Hamiltonian1,DensityMatrix1);

            cout << "Level 1 :" << endl;

            for (int number = 0; number < optN; ++number)
                {

                    Iso_env = ternary_Environment_Iso(Hamiltonian1,DensityMatrix2,Unitary1,Isometry1);
                    Ienvr = vector<Index> (Iso_env.indeces.begin(), Iso_env.indeces.begin()+3);
                    Ienvc = vector<Index> (Iso_env.indeces.begin()+3, Iso_env.indeces.begin()+4);
                    svd(U,s,V,Iso_env.toMat(Ienvr, Ienvc));
                    Isometry1.fromMat(-(V*U.submat(span(),span(0,cn1-1)).t()).st(),Ienvr, Ienvc);
                    //Isometry1.fromMat(-V.submat(span(),span(0,cn1-1))*U.t(),Ienvr, Ienvc);
                }
            Hamiltonian2 = ternary_Ascending(Hamiltonian1, Unitary1, Isometry1);
            cout << "energy at level 2 is :";
            energy(Hamiltonian2,DensityMatrix2);

            for (int number = 0; number < optN; ++number)
                {

                    Uni_env = ternary_Environment_Unit(Hamiltonian1,DensityMatrix2,Unitary1,Isometry1);
                    Uenvr = vector<Index> (Uni_env.indeces.begin(), Uni_env.indeces.begin()+2);
                    Uenvc = vector<Index> (Uni_env.indeces.begin()+2, Uni_env.indeces.begin()+4);
                    svd(U,s,V,Uni_env.toMat(Uenvr, Uenvc));
                    Unitary1.fromMat(-(V*U.t()).t(),Uenvr, Uenvc);
                }

            Hamiltonian2 = ternary_Ascending(Hamiltonian1, Unitary1, Isometry1);
            cout << "energy at level 2 is :";
            energy(Hamiltonian2,DensityMatrix2);

            cout << "Level 2 :" << endl;

            for (int number = 0; number < optN; ++number)
                {

                    Iso_env = ternary_Environment_Iso(Hamiltonian2,DensityMatrix2,Unitary2,Isometry2);
                    Ienvr = vector<Index> (Iso_env.indeces.begin(), Iso_env.indeces.begin()+3);
                    Ienvc = vector<Index> (Iso_env.indeces.begin()+3, Iso_env.indeces.begin()+4);
                    svd(U,s,V,Iso_env.toMat(Ienvr, Ienvc));
                    //Isometry2.fromMat(-V.submat(span(),span(0,cn2-1))*U.t(),Ienvr, Ienvc);
                    Isometry2.fromMat(-(V*U.submat(span(),span(0,cn2-1)).t()).st(),Ienvr, Ienvc);
                }

            for (int number = 0; number < optN; ++number)
                {

                    Uni_env = ternary_Environment_Unit(Hamiltonian2,DensityMatrix2,Unitary2,Isometry2);
                    Uenvr = vector<Index> (Uni_env.indeces.begin(), Uni_env.indeces.begin()+2);
                    Uenvc = vector<Index> (Uni_env.indeces.begin()+2, Uni_env.indeces.begin()+4);
                    svd(U,s,V,Uni_env.toMat(Uenvr, Uenvc));
                    Unitary2.fromMat(-(V*U.t()).st(),Uenvr, Uenvc);
                }
            Hamiltonian3 = ternary_Ascending(Hamiltonian2, Unitary2, Isometry2);

            cout << "recalculating Density Matrix" << endl;


            for (int i = 0; i < 8; ++i)
                {
                    cout << i+1 <<"\t";
                    DensityMatrix2 = ternary_Descending(DensityMatrix2, Unitary2, Isometry2);
                }
            cout << endl;
            Tensor DensityMatrix1 = ternary_Descending(DensityMatrix2,Unitary1,Isometry1);
            Tensor DensityMatrix0 = ternary_Descending(DensityMatrix1,Unitary0,Isometry0);

            cout << "energy at level 3 is :";
            energy(Hamiltonian3,DensityMatrix2);
            cout << "energy at level 2 is :";
            energy(Hamiltonian2,DensityMatrix2);
            cout << "energy at level 1 is :";
            energy(Hamiltonian1,DensityMatrix1);
            cout << "energy at level 0 is :";
            energy(Hamiltonian0,DensityMatrix0);

        }

}


//  LocalWords:  Isometry
