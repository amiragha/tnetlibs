#include <iostream>
#include "iDMRG.h"
#include <iomanip>
#include <cmath>

using namespace std;
using namespace arma;

/**
 * constructors
 */
IDMRG::IDMRG(int mD, double con_thresh){

    maxD = mD;
    converged = false;
    convergence_threshold = con_thresh;

    // introducing the Hamiltonian
    cx_mat PauliX,PauliY,PauliZ,I;
    PauliX << cx_d(0.0,0.0) << cx_d(1.0,0.0) << endr
           << cx_d(1.0,0.0) << cx_d(0.0,0.0) << endr;

    PauliY << cx_d(0.0,0.0) << cx_d(0.0,-1.0) << endr
           << cx_d(0.0,1.0) << cx_d(0.0,0.0) << endr;

    PauliZ << cx_d(1.0,0.0) << cx_d(0.0,0.0) << endr
           << cx_d(0.0,0.0) << cx_d(-1.0,0.0) << endr;

    cx_mat I2(2,2);
    I2.eye();

    // choosing model
    //string model = "heisenberg";
    string model = "kitaev";

    cx_mat matHamilt, matHamilt_lmost, matHamilt_rmost;
    if (model == "heisenberg"){
        // introducing the Heisenberg Hamiltonian matrices
        // TO-DO : find a better representation if possible
        matHamilt = zeros<cx_mat>(10,10);
        matHamilt.submat(0,0,1,1) = I2;
        matHamilt.submat(8,8,9,9) = I2;
        matHamilt.submat(2,0,3,1) = PauliX;
        matHamilt.submat(8,2,9,3) = PauliX;
        matHamilt.submat(4,0,5,1) = PauliY;
        matHamilt.submat(8,4,9,5) = PauliY;
        matHamilt.submat(6,0,7,1) = PauliZ;
        matHamilt.submat(8,6,9,7) = PauliZ;
        //matHamilt.submat(8,0,9,1) = -I2;

        cout << matHamilt << endl;
        matHamilt_lmost = matHamilt.rows(8,9);
        matHamilt_rmost = matHamilt.cols(0,1);

        // set the Hamiltonian and spin cardinalities
        B = 5;
        d = 2;
    }
    else if (model == "kitaev"){
        // introducing the ladder Kite Hamiltonian matrices
        /* every three particles is bundled into one particle
         */

	double J_pv = 1.0, J_cv = 0.5;
        cx_mat Sig_zzz, Sig_zII, Sig_IxI, Sig_xxI, Sig_IIx, Sig_xII,
            Sig_xIx, Sig_xzx, Sig_Ixz, Sig_zxI, eye8(8,8);
        Sig_zzz = kron(kron(PauliZ, PauliZ), PauliZ);
        Sig_zII = kron(kron(PauliZ, I2), I2);
        Sig_IxI = kron(kron(I2, PauliX), I2);
        Sig_xxI = kron(kron(PauliX, PauliX), I2);
        Sig_IIx = kron(kron(I2, I2), PauliX);
        Sig_xIx = kron(kron(PauliX, I2), PauliX);
        Sig_xzx = kron(kron(PauliX, PauliZ), PauliX);
        Sig_Ixz = kron(kron(I2, PauliX), PauliZ);
        Sig_zxI = kron(kron(PauliZ, PauliX), I2);
        Sig_xII = kron(kron(PauliX, I2), I2);

        eye8.eye();

        bool nocluster = true;
        if (nocluster){
            matHamilt = zeros<cx_mat>(40,40);
            matHamilt.submat(0,0,7,7) = eye8;
            matHamilt.submat(32,32,39,39) = eye8;
            matHamilt.submat(8,0,15,7) = Sig_zII;
            matHamilt.submat(32,8,39,15) = Sig_zzz;
            matHamilt.submat(16,0,23,7) = J_pv * Sig_xxI;
            matHamilt.submat(32,16,39,23) = Sig_IxI;
            matHamilt.submat(24,0,31,7) = J_pv * Sig_xIx;
            matHamilt.submat(32,24,39,31) = Sig_IIx;

            matHamilt_lmost = matHamilt.rows(32,39);
            matHamilt_rmost = matHamilt.cols(0,7);

            // set the Hamiltonian and spin cardinalities
            B = 5;
        }
        else {
            matHamilt = zeros<cx_mat>(56,56);
            matHamilt.submat(0,0,7,7) = eye8;
            matHamilt.submat(48,48,55,55) = eye8;
            matHamilt.submat(8,0,15,7) = Sig_zII;
            matHamilt.submat(48,8,55,15) = Sig_zzz;
            matHamilt.submat(16,0,23,7) = J_pv * Sig_xxI;
            matHamilt.submat(48,16,55,23) = Sig_IxI;
            matHamilt.submat(24,0,31,7) = J_pv * Sig_xIx;
            matHamilt.submat(48,24,55,31) = Sig_IIx;
            matHamilt.submat(32,0,39,7) = J_cv * Sig_xII;
            matHamilt.submat(48,32,55,39) = Sig_Ixz;
            matHamilt.submat(40,0,47,7) = J_cv * Sig_zxI;
            matHamilt.submat(48,40,55,47) = Sig_IIx;
            matHamilt.submat(48,0,55,7) = J_cv * Sig_xzx;


            matHamilt_lmost = matHamilt.rows(32,39);
            matHamilt_rmost = matHamilt.cols(0,7);

            // set the Hamiltonian and spin cardinalities
            B = 7;
        }

        d = 8;
    }
    /* the first dimension is always one
     * This is due to open boundary condition
     */
    // TO-DO : check for close boundary conditions
    matDims.push_back(1);

    // initialization of needed indeces for the whole class
    // TO-DO : find a better initialization for indeces
    bl.i("bl",B), br.i("br",B), b.i("b",B),
        sul.i("sul",d), sdl.i("sdl",d), sur.i("sur",d), sdr.i("sdr",d),
        sd.i("sd",d), su.i("su",d);


    /* Note:
     * here WL has Indeces = sdl:d, sul:d, b:B
     * here WR has Indeces = sdr:d, b:B, sur:d
     * then W  has Indeces = sdl:d, sul:d, sdr:d, sur:d
     */
    bool subtracting_largest_eigenvalue = true;
    if (subtracting_largest_eigenvalue){
        vec eigenvals;
        cx_mat eyed(d,d);
        eyed.eye();
        WL.fromMat(matHamilt_lmost, mkIdxSet(sdl), mkIdxSet(sul,b));
        WR.fromMat(matHamilt_rmost, mkIdxSet(sdr,b), mkIdxSet(sur));
        W = WL * WR;
        W.rearrange(mkIdxSet(sdl,sdr,sul,sur));
        eig_sym(eigenvals,W.toMat(2,2));
        largestEV = eigenvals(d*d - 1);
        matHamilt.submat(B*d-d,0,B*d-1,d-1) =
            matHamilt.submat(B*d-d,0,B*d-1,d-1) -largestEV * eyed;
        cout << matHamilt << endl;
        matHamilt_lmost = matHamilt.rows(B*d-d, B*d-1);
        matHamilt_rmost = matHamilt.cols(0,d-1);
    }
    else largestEV = 0.0;

    WL.fromMat(matHamilt_lmost, mkIdxSet(sdl), mkIdxSet(sul,b));
    WR.fromMat(matHamilt_rmost, mkIdxSet(sdr,b), mkIdxSet(sur));
    W = WL * WR;
    W.rearrange(mkIdxSet(sdl,sdr,sul,sur));

    //W.print(4);

    // defining Hamiltonian Tensors
    // constructing Hamiltonian Tensors from matrices
    Hamilt.fromMat(matHamilt, mkIdxSet(sd,bl), mkIdxSet(su,br));

    iteration = 0;
    // start the initial setup
    zeroth_iter(true);
    iterate();
}

IDMRG::~IDMRG(){

}

/**
 * lambda_size_trunc
 * given the vector of lambdas, it will check them for small values and
 * find the next Dimension : nextD
 *
 * param S vector of not yet truncated lambdas
 *
 * return int nextD
 */
int
IDMRG::lambda_size_trunc (const vec & S){
    double svalue;
    double threshold = 1.0e-9;
    int nextD = 0;
    for (nextD = 0; nextD < S.size(); ++nextD){
        svalue = S[nextD];
        if (svalue < 0.0)
            svalue = -svalue;
        if (svalue < threshold)
            break;
    }
    // nextD is now on it's proper value ( if not larger than maxD)
    if (nextD > maxD)
        nextD = maxD;

    cout << "next D is " << nextD << " maxD : " << maxD << endl;
    // check for potential mistakes like double inserts
    matDims.push_back(nextD);
    return nextD;
}

/**
 * zeroth_iter
 * managing the zeroth step which are solving for the
 * two site lattice (a part of the initialization step)
 */
void
IDMRG::zeroth_iter(bool verbose){
    if (verbose)
        cout << "Zeroth level starting" << endl;

    // getting the D from the matDims
    int D = matDims.back();
    int nextD;

    // solving the eigenvalue problem for W.toMat(2,2);
    cx_mat eigenvecs;
    vec eigenvals;
    eig_sym(eigenvals,eigenvecs,W.toMat(2,2));

    if (verbose)
        cout << "eigenvals: " << endl << eigenvals << endl;

    /*
     * Note :
     * here eigenvecs.col(0) is d.d vector
     * we have to reshape it into a d*d matrix
     */
    cx_mat U,V;
    vec S;
    svd(U,S,V,reshape(eigenvecs.col(0),d,d));

    // the starting energy for the 2site problem
    energy.push_back(eigenvals(0));
    convergence.push_back(1);

    // lambda size check
    nextD = lambda_size_trunc(S);

    if (verbose)
        cout << "nextD = " << nextD << "   ";

    // the truncation step
    cx_mat U_trunc = U.cols(0, nextD-1);
    cx_mat V_trunc = V.cols(0, nextD-1);
    vec S_trunc = S(span(0,nextD-1));

    // stroing the lambda
    // truncated lambda
    lambda_truncated.push_back(0);
    lambda.push_back(S_trunc);

    if (verbose)
        cout << "lambda 0 :" << endl << lambda[0] << endl;

    // updating Left and Right
    // for the zeroth level it happens here

    Index lu("lu",nextD), ld("ld",nextD), ru("ru",nextD), rd("rd",nextD);
    Tensor A, B, Astar, Bstar;
    A.fromMat(U_trunc,mkIdxSet(sul),mkIdxSet(lu));
    B.fromMat(V_trunc.t(), mkIdxSet(ru),mkIdxSet(sur));

    Astar = A.conjugate();
    Bstar = B.conjugate();
    Astar.reIndex(mkIdxSet(sdl,ld));
    Bstar.reIndex(mkIdxSet(rd,sdr));

    WL.reIndex(mkIdxSet(sdl, sul, bl));
    WR.reIndex(mkIdxSet(sdr,br,sur));
    Left = A * WL * Astar;
    Right = B * WR * Bstar;

    // guessing for the first level
    cx_mat guessmat = randu<cx_mat>(d*nextD,d*nextD);
    guess.fromMat(guessmat, mkIdxSet(sul,lu), mkIdxSet(sur,ru));

    /*
     * update WL, WR and W to their final value
     * Note:
     * after the following updating WL,WR and W will be fixed for ever
     * the first reIndexing of WL and WR is for constructing W
     * the second is for them to be appropriate for calculating Left and
     * Right
     * They finally have the indeces:
     *
     */
    WL = Hamilt;
    WR = Hamilt;
    WL.reIndex(mkIdxSet(sdl,bl,sul,b));
    WR.reIndex(mkIdxSet(sdr,b,sur,br));

    // W indeces : sdl:d, bl:B, sul:d, sdr:d, sur:d, br:B
    W = WL * WR;
    WL.reIndex(mkIdxSet(sdl,b,sul,bl));
    WR.reIndex(mkIdxSet(sdr,br,sur,b));

}

/**
 * do_step
 * go one step forward in the iDMRG algorithm
 */
void
IDMRG::do_step(bool verbose){
    cout << endl;
    iteration++;
    if (verbose)
        cout << "Starting iteration number : " << iteration << endl;

    int nextD, D = matDims.back();

    cx_mat ksiVec = Lanczos();
    //cout << "ksiVec" << endl << ksiVec << endl;

    Index lu("lu", D), ru("ru", D),ld("ld", D), rd("rd", D);

    /*
     * exact calculation for comparison,
     * indeces of HH = lu,ld,sdl,sul,sdr,sur,ru,rd
     cout << "left right indeces and w " << endl;
     Left.reIndex(mkIdxSet(lu,bl,ld));
     Right.reIndex(mkIdxSet(ru,br,rd));
     Tensor HH = Left * W * Right;
     cx_mat matHH = HH.toMat(mkIdxSet(sdl,ld,sdr,rd),
     mkIdxSet(sul,lu,sur,ru));
     cx_mat eigenvecs;
     vec eigenvals;
     eig_sym(eigenvals, eigenvecs, matHH);
     cout << setprecision(15) <<"groundstate energy = " <<
     eigenvals(0)/(8*(iteration+1)) << endl;
     cout << setprecision(15) << "fidelity of exact and lanczos :  " <<
     abs(cdot(eigenvecs.col(0),ksiVec)) << endl;
    */

    /*
     * Note:
     * here we received the ksiVec from lanczos
     * we suppose that the result of the lanczos vector has the following
     * indeces = sul:d ,lu:D ,sur:d ,ru;D
     */

    vector<Index> ksiInd = mkIdxSet(sul,lu,sur,ru);
    Tensor ksi(ksiInd);
    ksi.fromVec(ksiVec, ksiInd);

    cx_mat U,V;
    vec S;
    svd(U,S,V,ksi.toMat( mkIdxSet(lu,sul), mkIdxSet(ru,sur) ) );

    // lambda size check
    nextD = lambda_size_trunc(S);
    if (verbose)
        cout << "nextD = " << nextD << "   ";

    // truncation step
    cx_mat U_trunc = U.cols(0, nextD-1);
    cx_mat V_trunc = V.cols(0, nextD-1);
    vec S_trunc = S(span(0, nextD-1));

    // truncated lambda
    double lambda_norm = norm(S_trunc,2);
    lambda_truncated.push_back(1-lambda_norm);
    lambda.push_back(S_trunc/lambda_norm);

    mat S_trunc_mat = diagmat(S_trunc);

    if (verbose){
        //cout << "S" << endl << S << endl;
        cout << "lambda " << iteration+1 << endl << lambda.back() << endl;
    }

    // calculating guess for the next iteration
    guess_calculate(U_trunc, V_trunc, S_trunc_mat, D, nextD);

    update_LR(U_trunc, V_trunc, D, nextD);
    // if (!converged){
    //         // updating Left and Right
    //     }
    //     else {
    //         canonicalize();
    //     }
}

/**
 * guess_calculate
 * given the truncated and ready to use U, V, S(mat), will rotate the center
 * and correctly calculated a guess (trial) tensor for lanczos to use
 */
void IDMRG::guess_calculate(const cx_mat & U, const cx_mat & V,
                            const mat & S, int D, int nextD){
    // TO-DO : rotation check

    cx_mat newA, newB, left_lambda, right_lambda,u,v;
    mat diags;
    vec s;
    /*
     * rotate left
     * note: U_trunc * S_trunc is a matrix of D.d*nextD dimension
     * we need to perform svd on the matrix D*d.nextD
     *
     * and since only we have QR factorization we use it on lft.t() and
     * then find the Q.t() and R.t() to perform a RQ transition
     */
    cout << "starting left rotation!" << endl;
    cx_mat lft = (U * S);
    lft.reshape(D,d*nextD);
    svd(u, s, newB, lft);
    newB = newB.t();
    diags = diagmat(s);
    diags.resize(D,d*nextD);
    left_lambda = u*diags;

    /*
     * rotate right
     * note: S_trunc * V.t()_trunc is a matrix of nextD*D.d dimension
     * we need to perform svd on the matrix d.nextd*D
     */
    cout << "starting right rotation1" << endl;
    cx_mat rgt = (S * V.t()).st();
    rgt.reshape(D, d*nextD);
    svd(newA,s,v,rgt.st());
    diags = diagmat(s);
    diags.resize(nextD*d,D);
    right_lambda = diags*v.t();


    /*
     * note for rotations:
     * newA and newB are d.nextD * d.nextD matrices
     * right_lambda is d.nextD * D
     * left_lambda is D * d.nextD
     */

    /*
     * guess calculations
     * Mcculloch suggestion :
     * guess = newA * right_lambda * [lambda (l-1)]^-1 * left_lambda * newB
     * (d.nextD,d.nextD)*(d.nextD,D)*(D,D)*(D,d.nextd)*(d.nextd,d.nextd)
     * guesscore is the inverse of the last lambda
     */
    int n = lambda.size()-1;
    cout << "guess calculation" << endl;
    mat guesscore = inv(diagmat(lambda[n-1]));

    // convergence test
    vec singular;
    double cnvg_fidelity;
    if (iteration > 1){
        svd(singular, right_lambda * diagmat(lambda[n-1]));
        cnvg_fidelity = 1-sum(singular);
    }
    else
        cnvg_fidelity = 1;
    convergence.push_back(cnvg_fidelity);
    if (cnvg_fidelity < convergence_threshold)
        converged = true;

    cx_mat guessMat = newA * right_lambda * guesscore * left_lambda * newB;
    //cx_mat guessMat = randu<cx_mat>(d*nextD,d*nextD);

    // Defining new l and r indeces and
    Index nru("ru",nextD), nlu("lu",nextD);

    // building the Tensor guess
    guess.fromMat(guessMat,mkIdxSet(sul,nlu),mkIdxSet(sur,nru));

}

/**
 * update_LR
 * given the new A and B, updates Left and Right matrices
 */
void IDMRG::update_LR(const cx_mat & U, const cx_mat & V,
                      int D, int nextD){

    // indeces needed
    Index nlu("lu",nextD),nld("ld",nextD),nru("ru",nextD),nrd("rd",nextD);
    Index plu("plu",D), pld("pld",D), pru("pru",D), prd("prd",D);

    Tensor A, B, Astar, Bstar;
    A.fromMat(U,mkIdxSet(plu,sul),mkIdxSet(nlu));
    B.fromMat(V.t(), mkIdxSet(nru),mkIdxSet(pru,sur));
    Astar = A.conjugate();
    Bstar = B.conjugate();
    Astar.reIndex(mkIdxSet(pld,sdl,nld));
    Bstar.reIndex(mkIdxSet(nrd,prd,sdr));

    int n = lambda.size()-1;
    if (!converged){
        Left.reIndex(mkIdxSet(plu,b,pld));
        Right.reIndex(mkIdxSet(pru,b,prd));
        // constructing the Left and Right
        cout << "constructing Left and Right" << endl;
        Left = ((Left * A) * WL) * Astar;
        //Left.printIndeces();
        Right = ((Right * B) * WR) * Bstar;
        //Right.printIndeces();
    }
    else {
        // TO-DO : check for D == nextD
        if (D != nextD)
            cout << "WARNING: D != nextD in canonicalization" << endl;
        // canonicalization
        cout << "starting Canonilcalization" << endl;
        // according to mcculloch
        /*
         * Notations:
         * Lam : Lambda[n]
         * Lambar :inv(Lambda[n-1])
         */

        // needed Indexes and Tensors
        Index gmr("gmr", D), gml("gml", D), ou("ou", D), od("od", D),
            vu("vu", D), vd("vd", D), tl("tl", D), tr("tr",D), lu("lu",D), ru("ru", D);
        Tensor Lam, Lambar, Vec;
        cx_mat Vecmat, Vecvecs;
        cx_mat X, Y;
        vec Vecvals;
        cx_mat eyeD;
        eyeD.eye(D,D);
        cx_mat dlam,dlambar;
        mat old_lambda = diagmat(lambda[n-1]);
        dlam = eyeD * diagmat(lambda[n]);
        dlambar = eyeD * inv(diagmat(lambda[n-1]));
        Lam.fromMat(dlam,mkIdxSet(gml),mkIdxSet(gmr));

        // left canonicalization
        /*
         * indeces:
         * A = vu, sul, gml
         * B = grm, pru, sur
         * Lam = gml, gmr
         * Lambar = pru, ou
         */
        Lambar.fromMat(dlambar,mkIdxSet(pru), mkIdxSet(ou));
        A.reIndex(mkIdxSet(vu,sul,gml));
        B.reIndex(mkIdxSet(gmr,pru,sur));
        UP_tensor = A * Lam * B * Lambar;
        UP_tensor.printIndeces();
        DN_tensor = UP_tensor.conjugate();
        DN_tensor.reIndex(mkIdxSet(vd,sul,sur,od));
        Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
        cout << arnoldi_canonical(Vec) << endl;
        Vecmat = Vec.toMat(1,1);
        //Vecmat = (Vecmat + Vecmat.t())/2;
        Vecmat = Vecmat/Vecmat(0,0);
        eig_sym(Vecvals,Vecvecs,Vecmat);
        cout << "left : " << endl << Vecvals << endl;
        Y = Vecvecs*diagmat(sqrt(Vecvals));
        Y = Y.st();

        // right canonicalization
        A.reIndex(mkIdxSet(plu, sul, gml));
        B.reIndex(mkIdxSet(gmr, vu, sur));
        Lam.reIndex(mkIdxSet(gml,gmr));
        Lambar.reIndex(mkIdxSet(ou,plu));
        UP_tensor = Lambar * A * Lam * B;
        UP_tensor.printIndeces();
        DN_tensor = UP_tensor.conjugate();
        DN_tensor.reIndex(mkIdxSet(od,sul,vd,sur));
        Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
        cout << arnoldi_canonical(Vec) << endl;
        Vecmat = Vec.toMat(1,1);
        //Vecmat = (Vecmat + Vecmat.t())/2;
        Vecmat = Vecmat/Vecmat(0,0);
        eig_sym(Vecvals,Vecvecs,Vecmat);
        //eig_gen(iVecvals,Vecvecs,Vecmat);
        cout << "right : " << endl << Vecvals << endl;
        X = Vecvecs*diagmat(sqrt(Vecvals));

        cout << "defining new lambda and Gamma" << endl;
        // defining new lambda and new gamma
        vec new_lambda_vec;
        cx_mat U,V;
        // cout << "old_lambda" << old_lambda <<endl;
        // cout << "Y" << Y <<endl;
        // cout << "X" << X <<endl;
        svd(U, new_lambda_vec, V, (Y * old_lambda * X) );
        cx_mat templeft_mat, tempright_mat;
        Tensor templeft, tempright, rLambar, lLambar, new_Gamma, new_lambda;
        // cout << "V" << V << endl;
        rLambar = Lambar;
        lLambar = Lambar;
        A.reIndex(mkIdxSet(plu, sul, gml));
        B.reIndex(mkIdxSet(gmr, pru, sur));
        Lam.reIndex(mkIdxSet(gml, gmr));
        lLambar.reIndex(mkIdxSet(tl, plu));
        rLambar.reIndex(mkIdxSet(pru, tr));
        templeft_mat = V.t() * inv(X);
        cout << "1" << endl;
        templeft.fromMat(templeft_mat,mkIdxSet(lu),mkIdxSet(tl));
        tempright_mat = inv(Y) * U;
        tempright.fromMat(tempright_mat,mkIdxSet(tr),mkIdxSet(ru));
        new_Gamma = templeft *  lLambar * A * Lam * B * rLambar * tempright;
        new_Gamma.printIndeces();
        new_lambda.fromMat(eyeD * diagmat(new_lambda_vec), mkIdxSet(ru),mkIdxSet(ou));

        // checking canonicalization
        cout << "checking canonicalization" << endl;
        // right check
        cout << "right check" << endl;
        UP_tensor = new_Gamma * new_lambda;
        UP_tensor.reIndex(ou,sul,sur,vu);
        UP_tensor.printIndeces();
        DN_tensor = UP_tensor.conjugate();
        DN_tensor.reIndex(mkIdxSet(od,sul,sur,vd));
        Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
        cout << arnoldi_canonical(Vec) << endl;
        //Vec.print(20);
        Vecmat = Vec.toMat(1,1);
        Vecmat = Vecmat/Vecmat(0,0); // killing the irrelevant phase factor
        eig_sym(Vecvals,Vecmat);
        cout << "right check is : " << endl << Vecvals;

        // left check
        /*
         * find the product of lambda from left to Gamma and find the new
         * left largest eigenvalue and the corresponding eigenvector
         */
        cout << "left check" << endl;
        new_Gamma.reIndex(mkIdxSet(lu,sul,sur,ou));
        new_lambda.reIndex(mkIdxSet(vu,lu));
        // now UP_tensor must have Indexes: vu, sul, sur, ou
        UP_tensor = new_lambda * new_Gamma;
        DN_tensor = UP_tensor.conjugate();
        DN_tensor.reIndex(mkIdxSet(vd,sul,sur,od));
        Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
        cout << arnoldi_canonical(Vec) << endl;
        //Vec.print(20);
        Vecmat = Vec.toMat(1,1);
        Vecmat = Vecmat/Vecmat(0,0); // killing the irrelevant phase factor
        eig_sym(Vecvals,Vecmat);
        cout << "left check is : " << endl << Vecvals;

    }
}
/**
 * canonicalize
 * canonicalize the wavefunction using the middle A,B, lambda calculated
 */
void IDMRG::canonicalize(){

}

/**
 * operateH
 * find the effect of bigH(Hamiltonians and Left and Right) on a given
 * vector
 * converts it to Tensor first, performs the contraction and then gives the
 * result back as vector
 * input of operateH must be a vector built out of a tensor
 * with indeces = sul:d, lu:D, sur:d, ru:D
 */
cx_vec
IDMRG::operateH(cx_vec & q){
    // initial reIndexing
    // TO-DO : check for input problems

    // extracting the needed indeces form input
    int D = matDims.back();

    Index lu("lu",D), ld("ld", D), ru("ru",D), rd("rd",D);

    /*
     * W indeces = sdl:d, bl:B, sul:d, sdr:d, sur:d, br:B
     */
    Left.reIndex(mkIdxSet(lu,bl,ld));
    Right.reIndex(mkIdxSet(ru,br,rd));

    // given Tensor must have the indeces : sul, lu, sur, ru
    Tensor given;
    given.fromVec(q,mkIdxSet(sul, lu, sur, ru));

    Tensor result;
    result = ((Left * given) * W) * Right;

    /*
     * after this operation the restult have : ld sdl sdr rd
     * so a rearrangement is necessary so the result will be a vector
     * built from a tensor with indeces = sdl ld sdr rd
     */
    result.rearrange(mkIdxSet(sdl, ld, sdr, rd));

    cx_vec res = result.toVec();
    return res;
}

/**
 * Lanczos
 * given a guess vector and a reference ksi, updates the ksi to the
 * eigenvector with smallest eigenvalue of L*W*R
 * input guess Tensor and the ksi vector as reference
 *
 * return void
 */
cx_vec
IDMRG::Lanczos(){

    cout << "starting lanczos!" << endl;
    cx_vec r, trial, final;
    cx_mat Q;
    cx_vec q;
    mat T, eigenvecs;
    vec eigenvals;
    double error = 1.0;
    vector<double> alphas, betas;
    int i;
    // first round
    guess.printIndeces();
    r= guess.toVec();

    q = r/norm(r,2);
    trial = q;
    r = operateH(q);
    alphas.push_back(cdot(q,r).real());
    r = r - (alphas[0] * q);
    betas.push_back(norm(r,2));
    Q = q;
    T << alphas[0];

    i = 0;
    //rounds
    while (true){
        q = r/betas[i];
        r = operateH(q) - betas[i] * Q.col(i);
        alphas.push_back(cdot(q,r).real());
        r = r - alphas[i+1] * q;

        // adding re-orthogonalization
        r = r - Q * (Q.t() * r);

        betas.push_back(norm(r,2));
        Q = join_rows(Q,q);

        // constructing the T matrix
        T.resize(i+2,i+2);
        T(i+1,i+1) = alphas[i+1];
        T(i+1,i) = betas[i];
        T(i,i+1) = betas[i];

        // calculating the eigenvalues of T
        eig_sym(eigenvals, eigenvecs, T);

        // Error estimation and convergence test
        // beta * eigenvecs last row
        error = betas[i+1] * eigenvecs(i+1,0);
        if (error < 0.0)
            error = -error;
        if ( error< 1.0e-15)
            break;
        // increment i
        ++i;
    }
    // fidelity calculations
    cout << "error: " << error << " , number of steps : " << i+1 << endl;
    //final = Q*eigenvecs.col(i+1);
    final = Q*eigenvecs.col(0);
    //energy.push_back(eigenvals(i+1));
    energy.push_back(eigenvals(0));
    cx_d f = cdot(trial,final);
    cout << "trial \\cdot final"<< f << endl;
    double fid = 1 - sqrt(f.real()*f.real()+f.imag()*f.imag());
    fidelity.push_back(fid);
    return final;
}


/**
 * arnoldi_canonical
 * performs arnoldi algorithms using UP_tensor and DN_tensor
 * a part of canonicalization process
 *
 */
cx_d IDMRG::arnoldi_canonical(Tensor & V){

    Index vu = V.indeces[0];
    Index vd = V.indeces[1];
    Tensor Vtemp;
    cx_vec h, resV;
    vec errors;
    cx_vec r = V.toVec();
    cx_vec q = r/norm(r,2);
    cx_mat T, Q = q;
    int i = 0;
    double error = 1;
    double hbefore = 0.0;
    cx_mat eigenvecs;
    cx_vec eigenvals;
    uword sss;
    while (error > 1.0e-14){
        // operating UP DN V
        Vtemp.fromVec(Q.col(i),mkIdxSet(vu,vd));
        Vtemp = (Vtemp * UP_tensor) * DN_tensor;
        //Vtemp.printIndeces();
        r = Vtemp.toVec();
        // orthogonalization step
        h = Q.t() * r;
        r = r - Q * h;

        // creating the matrix T
        if (i == 0)
            T = h;
        else {
            T.resize(i+1,i+1);
            T(i,i-1) = hbefore;
            T.col(i) = h;
        }
        //cout << T << endl;
        hbefore = norm(r,2);
        if (abs(hbefore) < 1.0e-15)
            break;

        if (i > 100)
            break;
        // eigensolving T
        eig_gen(eigenvals, eigenvecs, T);
        // eigenvals are not ordered
        // find the largest abs eigenvalue

        // convergence
        // cout << "eigenvals" << endl;
        // cout << abs(eigenvals) << endl;
        // cout << "maximum eigenvalue is" << endl;
        vec absvals = abs(eigenvals);
        absvals.max(sss);
        // Cout << "with index : "<< sss << endl;
        // cout << "eigenvecs" << endl << eigenvecs << endl;
        // cout << i << endl;
        errors = abs(eigenvecs.row(i)).st() * hbefore;
        // cout << "errors" << endl;
        // cout << errors<< endl;
        // cout << "with error" << endl;
        // cout << errors(sss) << endl;
        error = errors(sss);
        //cout << error << endl;
        resV = Q * eigenvecs.col(sss);
        q = r/hbefore;
        Q = join_rows(Q,q);

        i++;
    }
    //cout << "result is :" << eigenvals(sss) << endl;
    V.fromVec(resV, mkIdxSet(vu,vd));
    cout << "finished in : " << i << "steps" << endl;
    return eigenvals(sss);
}

/**
 * iterate
 * iterate to the convergence
 */

void
IDMRG::iterate(){
    int N = 100;
    for (int iter=0; iter < N; ++iter){
        do_step(true);
        if (converged)
            break;
    }

    cout << endl;

    int num_bonds, num_particles;

    // printing energy, Fidelity, truncations, D results at each level
    cout << "ENERGY , fidelity, truncation, D results" << endl;
    for (int i = 0 ; i < energy.size(); ++i) {
        num_particles = 2*(i+1);
        cout << setprecision(10) << energy[i]/num_particles + largestEV << "\t\t";
        cout << setprecision(10) << fidelity[i] << "\t\t";
        cout << setprecision(10) << lambda_truncated[i] << "\t\t";
        cout << setprecision(10) << matDims[i]<< "\t";
        cout << setprecision(10) << convergence[i];
        cout << endl;
    }
    cout << endl;

}
