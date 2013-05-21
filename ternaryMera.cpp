#include "ternaryMera.h"

using namespace std;
using namespace arma;

/**
 * constructors
 */
TernaryMera::TernaryMera(cx_mat & Hmat, int icard, int fcard, bool verbose) {
    if (verbose)
        cout << "Starting intialization of TernaryMera : " << endl;

    // level cardinalities
    int c = icard;
    while (c < fcard) {
        cards.push_back(c);
        c = c*c*c;
    }
    cards.push_back(fcard);

    numlevels = cards.size(); // finding the number of levels
    if (verbose)
        cout << "The problem have " << numlevels << " levels." << endl;

    // finding the starting Isometries and Unitaries
    for (int l = 0; l < numlevels; ++l) {
        give_random_UniIso(l);
    }
    if (verbose)
        cout << "Initialization of Unitaries and Isometries finshed." << endl;

    // creating starting Hamiltoninan from matrix
    Index h1("h1",icard), h2("h2",icard), h3("h3",icard), h4("h4",icard);
    vector<Index> hr = {h1, h2};
    vector<Index> hc = {h3, h4};
    Tensor H;
    H.fromMat(Hmat,hr,hc);
    Hamiltonian.push_back(H);

    // ascending the Hamiltonian to all levels
    for (int l = 0; l < numlevels; ++l)
        ascend(l,verbose);
    if (verbose)
        cout << "Initialization of Hamiltonins finished." << endl;

    // finding a random DensityMatrix
    Tensor dummy;
    for (int l = 0; l < numlevels; ++l)
        DensityMatrix.push_back(dummy);
    give_random_density();
    arnoldi(verbose);

    // descending the DensityMatrix to all levels
    for (int l = numlevels-1; l > 0; --l)
        descend(l,verbose);
    if (verbose)
        cout << "Initialization of DensityMatrices finished." << endl;

    // calculating the starting energy
    energy(verbose);
}

TernaryMera::~TernaryMera(){

}

/**
 * give_random_UniIso
 * find a random Unitary and Isometry, called during initialization
 *
 * return void
 */
void
TernaryMera::give_random_UniIso(int level){
    // check if it's possible to find this level
    if (Unitary.size() != level) {
        cout << "ERROR: can't initialize this level Unitary" << endl;
        return;
    }
    if (Isometry.size() != level) {
        cout << "ERROR: can't initialize this level " << endl;
        return;
    }

    //srand(time(NULL));
    int in = cards[level], out;
    if (level > numlevels-2)
        out = cards[numlevels-1];
    else
        out = cards[level+1];

    Index u1("u1",in),u2("u2",in),u3("u3",in),u4("u4",in),
        t1("t1",in),t2("t2",in),t3("t3",in),t4("t4",out);

    // defining the Unitary and Isometry
    vector<Index> u_idcs;
    vector<Index> u_ridx;
    vector<Index> u_cidx;
    vector<Index> t_idcs;
    vector<Index> t_ridx;
    vector<Index> t_cidx;

    // using singular value decomposition
    cx_mat X = randu<cx_mat>(in*in*in,in*in);
    cx_mat Umat, Vmat;
    vec s;
    svd(Umat,s,Vmat,X);

    u_idcs = {u1 ,u2, u3, u4};
    u_ridx = {u1, u2};
    u_cidx = {u3, u4};
    t_idcs = {t1 ,t2, t3, t4};
    t_ridx = {t1, t2, t3};
    t_cidx = {t4};

    Tensor U, I;
    U.indeces = u_idcs;
    U.fromMat(Vmat,u_ridx,u_cidx);
    I.indeces = t_idcs;
    I.fromMat(Umat,t_ridx,t_cidx);
    Unitary.push_back(U);
    Isometry.push_back(I);

}

/** give_random_density
 * find a random DensityMatrix satisfying all the properties
 * called during initialization
 *
 * return void
 */
void TernaryMera::give_random_density(){

    int c = cards[numlevels-1];
    Index d1("d1",c),d2("d2",c),d3("d3",c),d4("d4",c);
    vector<Index> d_idcs = {d1 ,d2, d3, d4};
    vector<Index> dr1  = {d1};
    vector<Index> dc1  = {d2};
    vector<Index> dr2  = {d3};
    vector<Index> dc2  = {d4};
    vector<Index> dvec1 = {d1,d2};
    vector<Index> dvec2 = {d3,d4};

    cx_mat X = randu<cx_mat>(c,1);
    cx_mat U, V;
    vec s;
    svd(U,s,V,X);
    Tensor U1(dvec1), U2(dvec2);

    U1.fromMat(U,dr1,dc1);
    U2.fromMat(U,dr2,dc2);
    U2.conjugate();

    DensityMatrix[numlevels-1] = (U1*U2)/(double)c;

}

/**
 * ascend
 * performing ascending for the  Hamiltonian at level with corresponding
 * Isometries and Unitaries, changes the Hamiltonian at level+1
 *
 * param level an int showing the wanted level
 *
 * return void
 */
void TernaryMera::ascend (int level, bool verbose){
    if (verbose)
        cout << "performing ascendig on level " << level << endl;

    // checking that the Hamiltonian exist
    if (Hamiltonian.size() < level + 1)
        cout << "ERROR: Hamiltonian at level " << level << " does'nt exist" << endl;
    int lvl = level;
    // checking for larger levels
    if (level > numlevels-1) lvl = numlevels-1;
    // create a copy of all needed Tensors
    Tensor H   = Hamiltonian[level];
    Tensor U   = Unitary[lvl];
    Tensor US  = Unitary[lvl];
    US.conjugate();
    Tensor T1  = Isometry[lvl];
    Tensor T1S = Isometry[lvl];
    T1S.conjugate();
    Tensor T2  = T1;
    Tensor T2S = T1S;

    // finding the cardinalities
    int in = cards[lvl], out;
    if (lvl > cards.size()-2)
        out = cards[lvl];
    else
        out = cards[lvl+1];

    Tensor Temp; // for keeping the temporary results

    // needed inedeces
    Index o1("o1",out),o2("o2",out),
        o3("o3",out),o4("o4",out),
        u1("u1",in),u2("u2",in),
        u3("u3",in),u4("u4",in),
        t1("t1",in),t2("t2",in),
        t3("t3",in),t4("t4",in),
        t5("t5",in),t6("t6",in),
        t7("t7",in),t8("t8",in),t9("t1",in);

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
    T1S.reIndex(t1s_idcs);
    T1.reIndex(t1_idcs);
    H.reIndex(h_idcs);
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    T2S.reIndex(t2s_idcs);
    T2.reIndex(t2_idcs);

    Tensor Left = ((T1S * T1) * ((H * U) * US)) * (T2S * T2);

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
    T1S.reIndex(t1s_idcs);
    T1.reIndex(t1_idcs);
    H.reIndex(h_idcs);
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    T2S.reIndex(t2s_idcs);
    T2.reIndex(t2_idcs);

    Tensor Center = ((T1S * T1) * ((H * U) * US)) * (T2S * T2);

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
    T1S.reIndex(t1s_idcs);
    T1.reIndex(t1_idcs);
    H.reIndex(h_idcs);
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    T2S.reIndex(t2s_idcs);
    T2.reIndex(t2_idcs);

    Tensor Right = ((T1S * T1) * ((H * U) * US)) * (T2S * T2);

    vector<Index> output_Idxs =  {o1, o2, o3, o4};

    Temp = ((Right + Center) + Left)/3;
    Temp.rearrange(output_Idxs);

    // check to see whether level+1 hamiltonian exists
    if (Hamiltonian.size() > level + 1)
        Hamiltonian[level+1] = Temp;
    else
        Hamiltonian.push_back(Temp);

}

/**
 * descend
 * performing descending of the DensityMatrix at level with corresponding
 * Isometris and Unitaries, changes the DensityMatrix at level-1
 *
 * param level an int showing the wanted level
 *
 * return void
 */
void TernaryMera::descend (int level, bool verbose){
    if (verbose)
        cout << "performing descending on level " << level << endl;

    // checking that the Hamiltonian exist
    if (Hamiltonian.size() < level + 1)
        cout << "ERROR: Hamiltonian at level " << level << " does'nt exist" << endl;
    int lvl = level;
    int downlvl = level-1;
    // checking for larger levels
    if (level > numlevels-1) {
        lvl = numlevels-1;
        downlvl = lvl;
    }
    // create a copy of all needed Tensors
    Tensor D   = DensityMatrix[lvl];
    Tensor U   = Unitary[downlvl];
    Tensor US  = Unitary[downlvl];
    US.conjugate();
    Tensor T1  = Isometry[downlvl];
    Tensor T1S = Isometry[downlvl];
    T1S.conjugate();
    Tensor T2  = T1;
    Tensor T2S = T1S;

    // finding the cardinalities
    int in = cards[lvl];
    int out = cards[downlvl];

    Tensor Temp; // for keeping the temporary results

    // needed indeces
    Index o1("o1",out),o2("o2",out),
        o3("o3",out),o4("o4",out),
        u1("u1",out),u2("u2",out),
        u3("u3",out),u4("u4",out),u5("u5",out),
        t1("t1",out),t2("t2",out),
        t3("t3",out),t4("t4",out),
        i1("i1",in),i2("i2",in),
        i3("i3",in),i4("i4",in);

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
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    D.reIndex(d_idcs);
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Left = (T1 * T1S) * ((T2 * T2S) * D) * (U * US);
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
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    D.reIndex(d_idcs);
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Center = U * ((T1 * T1S) * ((T2 * T2S) * D)) * US;

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
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    D.reIndex(d_idcs);
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Right = (T1 * T1S) * ((T2 * T2S) * D) * (U * US);
    Right.rearrange(output_Idxs);

    Temp = ((Right + Center) + Left)/3;

    // check for the level
    DensityMatrix[downlvl] = Temp;


}

/**
 * iso_env
 * finds the environment of an Isometry at level using Hamiltonian and
 * Unitary at the same level and DensityMatrix at level+1
 *
 * param level an int
 *
 * return Tensor of environment for isometry
 */
Tensor TernaryMera::iso_env (int level, bool verbose){

}

/**
 * uni_env
 * finds the environment of a Unitary  at level using Hamiltonian and
 * Isometry at the same level and DensityMatrix at level+1
 *
 * param level an int
 *
 * return Tensor of environment for unitary
 */
Tensor TernaryMera::uni_env (int level, bool verbose){

}

/**
 * arnoldi
 * performing arnoldi algorithm for the descend operator and finding
 * the fixed point DensityMatrix, changes DensityMatrix at the last level
 *
 * return void
 */
void TernaryMera::arnoldi(bool verbose){
    if (verbose)
        cout << "performing arnoldi algorithm" << endl;

    for (int i = 0; i < 8; ++i)
        {
            // cout << endl;
            // cout << i+1 <<endl;
            // for (int j = 0; j < 10; ++j)
            //     cout << DensityMatrix2.values[100*j] << "\t";
            descend(numlevels);
        }
}

/**
 * energy
 * finds the energy at the wanted level using the Hamiltonin and
 * DensityMatrix at the level, if called without any arguments
 * calculates the energy at all levels
 *
 * param level or no argument
 *
 * return double or a vector of doubles
 */
double TernaryMera::energy (int level){
    int lvl = level;
    if (level > numlevels-1)
        lvl = numlevels-1;
    int c = cards[lvl];

    Tensor H = Hamiltonian[level];
    Tensor D = DensityMatrix[lvl];
    // reIndexing for the correct contraction
    Index e1("e1",c),e2("e2",c),e3("e3",c),e4("e4",c);
    vector<Index> enIdx = {e1,e2,e3,e4};
    vector<Index> enIdx2 = {e3,e4,e1,e2};
    H.reIndex(enIdx);
    D.reIndex(enIdx2);
    Tensor energy = H * D;

    // changing the indeces back to initial indeces
    return energy.values[0].real();
}

vector<double> TernaryMera::energy (bool verbose){
    if (verbose)
        cout << "Energy calculations :" << endl;
    bool allequal = true;
    vector<double> result;
    // calculating energy at each level and put it at result vector
    for (int l = 0; l < numlevels + 1; ++l)
        result.push_back(energy(l));

    // checking to see whether the energies are equal at all levels
    double delta;
    for (int i = 1; i < result.size(); ++i) {
        delta = result[i] - result[0];
        if (delta < 0)
            delta = -delta;
        if (delta > thresh) {
            cout << "WARNING: energies are not equal" << endl;
            allequal = false;
            break;
        }
    }

    // printing the energies in verbose mode or if there is warning
    if (verbose || !allequal) {
        for (int i = 0; i < result.size(); ++i)
            cout << "lvl "<< i <<": "<< result[i] << "\t";
        cout << endl;
    }

    return result;
}

/**
 * iso_update
 * updates the Isometry at level with Unitary and Hamiltonian at the same
 * level and DensityMatrix from level+1, changes the Isometry at level
 *
 * param level
 *
 * return void
 */
void TernaryMera::iso_update (int level, bool verbose){

}

/**
 * uni_update
 * updates the Unitary at level with Isometry and Hamiltonian at the same
 * level and DensityMatrix from level+1, changes the Isometry at level
 *
 * param level
 *
 * return void
 */
void TernaryMera::uni_update (int level, bool verbose){

}

/**
 * bottom_up
 * performs one bottom_up iteration
 * updating isometries first then Unitries at each level then ascends
 * Hamiltonian at the end calls arnoldi on the last DensityMatrix and
 * descend the DensityMatrix to level 0
 * changes Unitaries, Isometries, Hamiltonians, DensityMatrices
 *
 * return void
 */
void TernaryMera::bottom_up(bool verbose){

}















// void ternaryMera (Tensor & Hamiltonian0, int max_card){

//     // note that max_card must be more than initial c*c*c
//     // extracting Hamiltonian indeces
//     vector<Index> h_input = Hamiltonian0.indeces;

//     Tensor Iso_env, Uni_env;
//     cout << "performing the optimization" << endl;
//     vector<Index> Ienvr;
//     vector<Index> Ienvc;
//     vector<Index> Uenvr;
//     vector<Index> Uenvc;
//     cx_mat U, V;
//     vec s;
//     int optN = 3;
//     // iteration loop
//     for (int iter = 0; iter < 10; ++iter)
//         {
//             cout << endl << "starting iteration number : " << iter+1 << " ==> "<< endl;
//             cout <<         "------------------------------" << endl;
//             cout << "Level 0 :" << endl;

//             for (int number = 0; number < optN; ++number)
//                 {
//                     Iso_env = ternary_Environment_Iso(Hamiltonian0,DensityMatrix1,Unitary0,Isometry0);
//                     Ienvr = vector<Index> (Iso_env.indeces.begin(), Iso_env.indeces.begin()+3);
//                     Ienvc = vector<Index> (Iso_env.indeces.begin()+3, Iso_env.indeces.begin()+4);
//                     svd(U,s,V,Iso_env.toMat(Ienvr, Ienvc));
//                     Isometry0.fromMat(-(V*U.submat(span(),span(0,cn0-1)).t()).st(),Ienvr, Ienvc);
//                     Hamiltonian1 = ternary_Ascending(Hamiltonian0, Unitary0, Isometry0);
//                     cout << "energy at level 1 is :";
//                     energy(Hamiltonian1,DensityMatrix1);
//                 }
//             Hamiltonian1 = ternary_Ascending(Hamiltonian0, Unitary0, Isometry0);
//             cout << "energy at level 1 is :";
//             energy(Hamiltonian1,DensityMatrix1);

//             for (int number = 0; number < optN; ++number)
//                 {
//                     Uni_env = ternary_Environment_Unit(Hamiltonian0,DensityMatrix1,Unitary0,Isometry0);
//                     Uenvr = vector<Index> (Uni_env.indeces.begin(), Uni_env.indeces.begin()+2);
//                     Uenvc = vector<Index> (Uni_env.indeces.begin()+2, Uni_env.indeces.begin()+4);
//                     svd(U,s,V,Uni_env.toMat(Uenvr, Uenvc));
//                     Unitary0.fromMat(-(V*U.t()).st(),Uenvr, Uenvc);
//                 }
//             Hamiltonian1 = ternary_Ascending(Hamiltonian0, Unitary0, Isometry0);
//             cout << "energy at level 1 is :";
//             energy(Hamiltonian1,DensityMatrix1);

//             cout << "Level 1 :" << endl;

//             for (int number = 0; number < optN; ++number)
//                 {

//                     Iso_env = ternary_Environment_Iso(Hamiltonian1,DensityMatrix2,Unitary1,Isometry1);
//                     Ienvr = vector<Index> (Iso_env.indeces.begin(), Iso_env.indeces.begin()+3);
//                     Ienvc = vector<Index> (Iso_env.indeces.begin()+3, Iso_env.indeces.begin()+4);
//                     svd(U,s,V,Iso_env.toMat(Ienvr, Ienvc));
//                     Isometry1.fromMat(-(V*U.submat(span(),span(0,cn1-1)).t()).st(),Ienvr, Ienvc);
//                     //Isometry1.fromMat(-V.submat(span(),span(0,cn1-1))*U.t(),Ienvr, Ienvc);
//                 }
//             Hamiltonian2 = ternary_Ascending(Hamiltonian1, Unitary1, Isometry1);
//             cout << "energy at level 2 is :";
//             energy(Hamiltonian2,DensityMatrix2);

//             for (int number = 0; number < optN; ++number)
//                 {

//                     Uni_env = ternary_Environment_Unit(Hamiltonian1,DensityMatrix2,Unitary1,Isometry1);
//                     Uenvr = vector<Index> (Uni_env.indeces.begin(), Uni_env.indeces.begin()+2);
//                     Uenvc = vector<Index> (Uni_env.indeces.begin()+2, Uni_env.indeces.begin()+4);
//                     svd(U,s,V,Uni_env.toMat(Uenvr, Uenvc));
//                     Unitary1.fromMat(-(V*U.t()).t(),Uenvr, Uenvc);
//                 }

//             Hamiltonian2 = ternary_Ascending(Hamiltonian1, Unitary1, Isometry1);
//             cout << "energy at level 2 is :";
//             energy(Hamiltonian2,DensityMatrix2);

//             cout << "Level 2 :" << endl;

//             for (int number = 0; number < optN; ++number)
//                 {

//                     Iso_env = ternary_Environment_Iso(Hamiltonian2,DensityMatrix2,Unitary2,Isometry2);
//                     Ienvr = vector<Index> (Iso_env.indeces.begin(), Iso_env.indeces.begin()+3);
//                     Ienvc = vector<Index> (Iso_env.indeces.begin()+3, Iso_env.indeces.begin()+4);
//                     svd(U,s,V,Iso_env.toMat(Ienvr, Ienvc));
//                     //Isometry2.fromMat(-V.submat(span(),span(0,cn2-1))*U.t(),Ienvr, Ienvc);
//                     Isometry2.fromMat(-(V*U.submat(span(),span(0,cn2-1)).t()).st(),Ienvr, Ienvc);
//                 }

//             for (int number = 0; number < optN; ++number)
//                 {

//                     Uni_env = ternary_Environment_Unit(Hamiltonian2,DensityMatrix2,Unitary2,Isometry2);
//                     Uenvr = vector<Index> (Uni_env.indeces.begin(), Uni_env.indeces.begin()+2);
//                     Uenvc = vector<Index> (Uni_env.indeces.begin()+2, Uni_env.indeces.begin()+4);
//                     svd(U,s,V,Uni_env.toMat(Uenvr, Uenvc));
//                     Unitary2.fromMat(-(V*U.t()).st(),Uenvr, Uenvc);
//                 }
//             Hamiltonian3 = ternary_Ascending(Hamiltonian2, Unitary2, Isometry2);

//             cout << "recalculating Density Matrix" << endl;


//             for (int i = 0; i < 8; ++i)
//                 {
//                     cout << i+1 <<"\t";
//                     DensityMatrix2 = ternary_Descending(DensityMatrix2, Unitary2, Isometry2);
//                 }
//             cout << endl;
//             Tensor DensityMatrix1 = ternary_Descending(DensityMatrix2,Unitary1,Isometry1);
//             Tensor DensityMatrix0 = ternary_Descending(DensityMatrix1,Unitary0,Isometry0);

//             cout << "energy at level 3 is :";
//             energy(Hamiltonian3,DensityMatrix2);
//             cout << "energy at level 2 is :";
//             energy(Hamiltonian2,DensityMatrix2);
//             cout << "energy at level 1 is :";
//             energy(Hamiltonian1,DensityMatrix1);
//             cout << "energy at level 0 is :";
//             energy(Hamiltonian0,DensityMatrix0);

//         }

// }


//  LocalWords:  Isometry
