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
    if (verbose)
        cout << "calculating the Iso environment for level " << level << endl;

    int uplevel = level + 1;
    // checking level
    if (level > numlevels-1)
        cout << "ERROR: level dos'nt exist" << endl;
    if (level > numlevels-2)
        uplevel = level;

    // extracting index information about input Tensors
    int in = cards[level];
    int out = cards[uplevel];

    // check for cardinality mismatch TO-DO

    // copy needed matrices
    Tensor D   = DensityMatrix[uplevel];
    Tensor H   = Hamiltonian[level];
    Tensor U   = Unitary[level];
    Tensor US  = Unitary[level];
    US.conjugate();
    Tensor T1  = Isometry[level];
    Tensor T1S = Isometry[level];
    T1S.conjugate();
    Tensor T2  = T1;
    Tensor T2S = T1S;

    Tensor Temp; // for keeping the temporary results

    // check for cardinality mismatch TO-DO

    // needed inedeces
    Index o1("o1",in),o2("o2",in),
        o3("o3",in),o4("o4",out),
        u1("u1",in),u2("u2",in),
        u3("u3",in),u4("u4",in),
        u5("u5",in),u6("u6",in),u7("u7",in),
        t1("t1",in),t2("t2",in),t3("t3",in),
        d1("d1",out),d2("d2",out),d3("d3",out);

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
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    D.reIndex(d_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Left_T1 = T1S * ((U * H * US) * (T2 * T2S) * D);
    Left_T1.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // T.reIndex(output_Idxs);
    // (Left_T1 * T).print(1);

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
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    D.reIndex(d_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Center_T1 = T1S * ((U * H * US) * (T2 * T2S) * D);
    Center_T1.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // T.reIndex(output_Idxs);
    // (Center_T1 * T).print(1);

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
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    D.reIndex(d_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Right_T1 = T1S * ((U * H * US) * (T2 * T2S) * D);
    Right_T1.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // T.reIndex(output_Idxs);
    // (Right_T1 * T).print(1);

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
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);
    D.reIndex(d_idcs);
    T2S.reIndex(t2s_idcs);

    Tensor Left_T2 = ((T1 * T1S) * (U * H * US) * D) * T2S;
    Left_T2.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // T.reIndex(output_Idxs);
    // (Left_T2 * T).print(1);

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
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);
    D.reIndex(d_idcs);
    T2S.reIndex(t2s_idcs);

    Tensor Center_T2 = ((T1 * T1S) * (U * H * US) * D) * T2S;
    Center_T2.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // T.reIndex(output_Idxs);
    // (Center_T2 * T).print(1);

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
    U.reIndex(u_idcs);
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);
    D.reIndex(d_idcs);
    T2S.reIndex(t2s_idcs);

    Tensor Right_T2 = ((T1 * T1S) * (U * H * US) * D) * T2S;
    Right_T2.rearrange(output_Idxs);
    // cout << "finished" << endl;
    // T.reIndex(output_Idxs);
    // (T * Right_T2).print(1);

    return (Left_T2 + Center_T2 + Right_T2 + Left_T1 + Center_T1 + Right_T1)/6;
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
    if (verbose)
        cout << "calculating the Uni environment for level " << level << endl;

    int uplevel = level + 1;
    // checking level
    if (level > numlevels-1)
        cout << "ERROR: level dos'nt exist" << endl;
    if (level > numlevels-2)
        uplevel = level;

    // extracting index information about input Tensors
    int in = cards[level];
    int out = cards[uplevel];

    // check for cardinality mismatch TO-DO

    // copy needed matrices
    Tensor D   = DensityMatrix[uplevel];
    Tensor H   = Hamiltonian[level];
    Tensor US  = Unitary[level];
    US.conjugate();
    Tensor T1  = Isometry[level];
    Tensor T1S = Isometry[level];
    T1S.conjugate();
    Tensor T2  = T1;
    Tensor T2S = T1S;

    Tensor Temp; // for keeping the temporary results

    // needed inedeces
    Index o1("o1",in),o2("o2",in),
        o3("o3",in),o4("o4",in),
        u1("u1",in),u2("u2",in),
        u3("u3",in),u4("u4",in),
        t1("t1",in),t2("t2",in),
        t3("t3",in),t4("t4",in),t5("t5",in),
        d1("d1",out),d2("d2",out),
        d3("d3",out),d4("d4",out);

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
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    D.reIndex(d_idcs);
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Left = (((T1 * T1S) * ((T2 * T2S) * D)) * US) * H;
    Left.rearrange(output_Idxs);

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
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    D.reIndex(d_idcs);
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Center = (T1 * T1S) * ((T2 * T2S) * D) * US * H;
    Center.rearrange(output_Idxs);

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
    US.reIndex(us_idcs);
    H.reIndex(h_idcs);
    D.reIndex(d_idcs);
    T2.reIndex(t2_idcs);
    T2S.reIndex(t2s_idcs);
    T1.reIndex(t1_idcs);
    T1S.reIndex(t1s_idcs);

    Tensor Right = (((T1 * T1S) * ((T2 * T2S) * D)) * US) * H;
    Right.rearrange(output_Idxs);

    return (Left + Center + Right)/3;
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
 * param num_update number of consecutive updates
 *
 * return void
 */
void TernaryMera::iso_update (int level, int num_update, bool verbose){
    if (verbose)
        cout << "updating Isometry of level " << level << endl;

    int uplevel = level+1;
    if (level > numlevels-2)
        uplevel = level;
    vector<Index> Ienvr;
    vector<Index> Ienvc;
    Tensor env;
    cx_mat U, V;
    vec s;
    for (int update = 0; update < num_update; ++update) {
        env = iso_env(level);
        Ienvr = vector<Index> (env.indeces.begin(), env.indeces.begin()+3);
        Ienvc = vector<Index> (env.indeces.begin()+3, env.indeces.begin()+4);
        svd(U,s,V,env.toMat(Ienvr, Ienvc));
        Isometry[level].fromMat(-(V*U.submat(span(),span(0,cards[uplevel]-1)).t()).st(),Ienvr, Ienvc);

    }
    // after the update, Hamiltonian at level+1 and DensityMatrix at level
    // must get updated as well.
    ascend(level);
    if (verbose)
        cout << energy(level+1) << endl;
}

/**
 * uni_update
 * updates the Unitary at level with Isometry and Hamiltonian at the same
 * level and DensityMatrix from level+1, changes the Isometry at level
 *
 * param level
 * param num_update number of consecutive updates
 *
 * return void
 */
void TernaryMera::uni_update (int level, int num_update, bool verbose){
    if (verbose)
        cout << "updating Unitary of level " << level << endl;

    vector<Index> Uenvr;
    vector<Index> Uenvc;
    Tensor env;
    cx_mat U, V;
    vec s;
    for (int update = 0; update < num_update; ++update) {
        env = uni_env(level);
        Uenvr = vector<Index> (env.indeces.begin(), env.indeces.begin()+2);
        Uenvc = vector<Index> (env.indeces.begin()+2, env.indeces.begin()+4);
        svd(U,s,V,env.toMat(Uenvr, Uenvc));
        Unitary[level].fromMat(-(V*U.t()).st(),Uenvr, Uenvc);
    }
    ascend(level);
    if (verbose)
        cout << energy(level+1) << endl;
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
    if (verbose)
        cout << "starting one bottom_up step" << endl;

    for (int i = 0; i < 3; ++i) {
        cout << i << endl;
        for (int l = 0; l < numlevels; ++l) {
            iso_update(l,10,verbose);
            uni_update(l,10,verbose);
        }
        arnoldi(verbose);
        for (int l = numlevels-1; l > 0; --l)
            descend(l);
        energy(verbose);
    }
}
