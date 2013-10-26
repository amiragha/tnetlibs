#include <iostream>
#include "iDMRG.h"
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace std;
using namespace arma;

/**
 * constructors
 */
IDMRG::IDMRG(cx_mat & mHamilt, u_int Bdim, u_int dim, u_int mD,
             double con_thresh, bool in_verbose, string logfile){

    verbose = in_verbose;
    B = Bdim;
    d = dim;
    maxD = mD;
    converged = false;
    convergence_threshold = con_thresh;

    // checking the sizes of the hamiltonian with the given Bdim and dim
    assert (mHamilt.size() == d * Bdim * Bdim * d);

    cx_mat matHamilt, matHamilt_lmost, matHamilt_rmost;
    matHamilt = mHamilt;
    matHamilt_lmost = matHamilt.rows(B*d-d, B*d-1);
    matHamilt_rmost = matHamilt.cols(0,d-1);

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
        matHamilt_lmost = matHamilt.rows(B*d-d, B*d-1);
        matHamilt_rmost = matHamilt.cols(0,d-1);
    }
    else largestEV = 0.0;

    WL.fromMat(matHamilt_lmost, mkIdxSet(sdl), mkIdxSet(sul,b));
    WR.fromMat(matHamilt_rmost, mkIdxSet(sdr,b), mkIdxSet(sur));
    W = WL * WR;
    W.rearrange(mkIdxSet(sdl,sdr,sul,sur));

    // defining Hamiltonian Tensors
    // constructing Hamiltonian Tensors from matrices
    Hamilt.fromMat(matHamilt, mkIdxSet(sd,bl), mkIdxSet(su,br));

    iteration = 0;

    lfout.open(logfile.c_str());
    // start the initial setup
    zeroth_iter();
    iterate();
    lfout.close();
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
u_int
IDMRG::lambda_size_trunc (const vec & S){
    double svalue;
    double threshold = 1.0e-15;
    u_int nextD = 0;
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

    if (verbose)
        lfout << "nexnt D is " << nextD << " maxD : " << maxD << endl;

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
IDMRG::zeroth_iter(){
    if (verbose)
        lfout << "Zeroth level starting" << endl;

    int nextD;

    // solving the eigenvalue problem for W.toMat(2,2);
    cx_mat eigenvecs;
    vec eigenvals;
    eig_sym(eigenvals,eigenvecs,W.toMat(2,2));

    if (verbose)
        lfout << "eigenvals: " << endl << eigenvals << endl;

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
        lfout << "nextD = " << nextD << "   ";

    // the truncation step
    cx_mat U_trunc = U.cols(0, nextD-1);
    cx_mat V_trunc = V.cols(0, nextD-1);
    vec S_trunc = S(span(0,nextD-1));

    // stroing the lambda
    // truncated lambda
    lambda_truncated.push_back(0);
    lambda.push_back(S_trunc);

    if (verbose)
        lfout << "lambda 0 :" << endl << lambda[0] << endl;

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
IDMRG::do_step(){
    iteration++;
    if (verbose){
        lfout << endl;
        lfout << "Starting iteration number : " << iteration << endl;
    }

    u_int nextD, D = matDims.back();

    cx_mat ksiVec = Lanczos();

    Index lu("lu", D), ru("ru", D),ld("ld", D), rd("rd", D);

    /*
     * exact calculation for comparison,
     * indeces of HH = lu,ld,sdl,sul,sdr,sur,ru,rd
     lfout << "left right indeces and w " << endl;
     Left.reIndex(mkIdxSet(lu,bl,ld));
     Right.reIndex(mkIdxSet(ru,br,rd));
     Tensor HH = Left * W * Right;
     cx_mat matHH = HH.toMat(mkIdxSet(sdl,ld,sdr,rd),
     mkIdxSet(sul,lu,sur,ru));
     cx_mat eigenvecs;
     vec eigenvals;
     eig_sym(eigenvals, eigenvecs, matHH);
     lfout << setprecision(15) <<"groundstate energy = " <<
     eigenvals(0)/(8*(iteration+1)) << endl;
     lfout << setprecision(15) << "fidelity of exact and lanczos :  " <<
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
        lfout << "nextD = " << nextD << "   ";

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
        lfout << "lambda " << iteration+1 << endl << lambda.back() << endl;
    }

    // calculating guess for the next iteration
    guess_calculate(U_trunc, V_trunc, S_trunc_mat, D, nextD);

    update_LR(U_trunc, V_trunc, D, nextD);

}

/**
 * guess_calculate
 * given the truncated and ready to use U, V, S(mat), will rotate the center
 * and correctly calculated a guess (trial) tensor for lanczos to use
 */
void IDMRG::guess_calculate(const cx_mat & U, const cx_mat & V,
                            const mat & S, u_int D, u_int nextD){
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
    if (verbose)
        lfout << "starting left rotation!" << endl;

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
    if (verbose)
        lfout << "starting right rotation!" << endl;

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
    if (verbose)
        lfout << "guess calculation" << endl;
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
    // if a fully random is needed
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
                      u_int D, u_int nextD){

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

    if (!converged && iteration < 100){
        Left.reIndex(mkIdxSet(plu,b,pld));
        Right.reIndex(mkIdxSet(pru,b,prd));
        // constructing the Left and Right
        if (verbose)
            lfout << "constructing Left and Right" << endl;
        Left = ((Left * A) * WL) * Astar;
        Right = ((Right * B) * WR) * Bstar;
    }
    else canonicalize(A, B, D, nextD);

}
/**
 * canonicalize
 * canonicalize the wavefunction using the middle A,B, lambda calculated
 */
void IDMRG::canonicalize(Tensor A, Tensor B, u_int D, u_int nextD){
    // canonicalization
    if (verbose)
        lfout << "starting Canonilcalization" << endl;

    u_int n = lambda.size()-1;

    /*
     * A and B indeces:
     * A := plu:D, sul:d, nlu:nextD
     * B := nru:D, pru:nextD, sur:d
     *
     */
    Index plu = A.indeces[0];
    Index sul = A.indeces[1];
    Index nlu = A.indeces[2];
    Index nru = B.indeces[0];
    Index pru = B.indeces[1];
    Index sur = B.indeces[2];
    /*
     * Justify truncation:
     * in the process of canonicalization D and nextD must be equal, so
     * if one is larger then truncation is necessary. In addition, if some
     * of the eigen values of right and left eigenvectors of the transfer
     * matrix is zero then we have to further reduce the dimension of the
     * matrices.
     */
    // if D > nextD
    if (D > nextD){
        if (verbose){
            lfout << "D = " << D << " is larger than nextD = " << nextD
                  << ", trunrcation has to occur" << endl;
        }
        // reduce D to nextD
        D = nextD;
        // A and B truncation
        cx_mat tempA, tempB;
        // using slice
        A = A.slice(plu,0,D);
        B = B.slice(pru,0,D);
        pru.card = D;
        plu.card = D;
    }

    // if D < nextD
    if (D != nextD && verbose)
        lfout << "WARNING: D != nextD in canonicalization" << endl;

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
    mat old_lambda = diagmat(lambda[n-1](span(0,D-1)));
    dlambar = eyeD * inv(diagmat(lambda[n-1](span(0,D-1))));
    dlam = eyeD * diagmat(lambda[n](span(0,D-1)));
    Lam.fromMat(dlam,mkIdxSet(gml),mkIdxSet(gmr));

    // left canonicalization
    /*
     * indeces:
     * A = vu, sul, gml
     * B = grm, pru, sur
     * Lam = gml, gmr
     * Lambar = pru, ou
     */
    // printing gammas
    //if (verbose){
    // A.printIndeces();
    // B.printIndeces();
    // lfout << "A :" << endl << A.toMat(mkIdxSet(plu,sul),mkIdxSet(nlu)) << endl;
    // lfout << "B :" << endl << B.toMat(mkIdxSet(nru,sur),mkIdxSet(pru)) << endl;
    //}
    cx_d arnoldi_res;
    Lambar.fromMat(dlambar,mkIdxSet(pru), mkIdxSet(ou));
    A.reIndex(mkIdxSet(vu,sul,gml));
    B.reIndex(mkIdxSet(gmr,pru,sur));
    UP_tensor = A * Lam * B * Lambar;
    DN_tensor = UP_tensor.conjugate();
    DN_tensor.reIndex(mkIdxSet(vd,sul,sur,od));
    Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
    arnoldi_res = arnoldi(Vec,UP_tensor,DN_tensor);
    if (verbose)
        lfout << arnoldi_res << endl;
    Vecmat = Vec.toMat(1,1);
    Vecmat = Vecmat/Vecmat.max();
    eig_sym(Vecvals,Vecvecs,Vecmat);
    if (verbose)
        lfout << "left : " << endl << Vecvals << endl;
    Y = Vecvecs*diagmat(sqrt(Vecvals));
    Y = Y.st();

    // the eigenvalue of sorted in ascending order
    u_int lastD_left;
    for (lastD_left = 0; lastD_left < Vecvals.size(); ++lastD_left){
        if (abs(Vecvals[lastD_left]) > 1.0e-10)
            break;
    }
    lastD_left = D - lastD_left;

    // right canonicalization
    A.reIndex(mkIdxSet(plu, sul, gml));
    B.reIndex(mkIdxSet(gmr, vu, sur));
    Lam.reIndex(mkIdxSet(gml,gmr));
    Lambar.reIndex(mkIdxSet(ou,plu));
    UP_tensor = Lambar * A * Lam * B;
    DN_tensor = UP_tensor.conjugate();
    DN_tensor.reIndex(mkIdxSet(od,sul,vd,sur));
    Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
    arnoldi_res = arnoldi(Vec,UP_tensor,DN_tensor);
    if (verbose)
        lfout << arnoldi_res << endl;
    Vecmat = Vec.toMat(1,1);
    Vecmat = Vecmat/Vecmat.max();
    eig_sym(Vecvals,Vecvecs,Vecmat);
    //eig_gen(iVecvals,Vecvecs,Vecmat);
    if (verbose)
        lfout << "right : " << endl << Vecvals << endl;
    X = Vecvecs*diagmat(sqrt(Vecvals));

    u_int lastD_right;
    for (lastD_right = 0; lastD_right < Vecvals.size(); ++lastD_right){
        if (abs(Vecvals[lastD_right]) > 1.0e-10)
            break;
    }
    lastD_right = D - lastD_right;

    // check for compatibility of left and right
    assert (lastD_right == lastD_left);
    u_int lastD = lastD_right;
    //int lastD = 2;
    if (lastD != D){
        if (verbose)
            lfout << "D is : " << D << " and lastD is : " << lastD << endl;
        // performing needed truncaton
        A.reIndex(mkIdxSet(plu, sul, nlu));
        B.reIndex(mkIdxSet(nru, pru, sur));
        A = A.slice(plu,0,lastD).slice(nlu,0,lastD);
        B = B.slice(nru,0,lastD).slice(pru,0,lastD);
        canonicalize(A,B,lastD,lastD);
        return;

    } else
        lastD = D;

    if (verbose){
        lfout << "old Von-Neuman entropy is : " << renyi(1.0,lambda[n-1])<< endl;
        lfout << "defining new lambda and Gamma" << endl;
    }
    // defining new lambda and new gamma
    vec new_lambda_vec;
    cx_mat U,V;
    svd(U, new_lambda_vec, V, (Y * old_lambda * X) );
    // normalizing new_lambda_vec
    new_lambda_vec = new_lambda_vec/norm(new_lambda_vec,2);

    cx_mat templeft_mat, tempright_mat;
    Tensor templeft, tempright, rLambar, lLambar, new_Gamma, new_lambda;
    rLambar = Lambar;
    lLambar = Lambar;
    A.reIndex(mkIdxSet(plu, sul, gml));
    B.reIndex(mkIdxSet(gmr, pru, sur));
    Lam.reIndex(mkIdxSet(gml, gmr));
    lLambar.reIndex(mkIdxSet(tl, plu));
    rLambar.reIndex(mkIdxSet(pru, tr));
    templeft_mat = V.t() * inv(X);
    templeft.fromMat(templeft_mat,mkIdxSet(lu),mkIdxSet(tl));
    tempright_mat = inv(Y) * U;
    tempright.fromMat(tempright_mat,mkIdxSet(tr),mkIdxSet(ru));
    new_Gamma = templeft *  lLambar * A * Lam * B * rLambar * tempright;
    //new_Gamma.printIndeces();
    new_lambda.fromMat(eyeD * diagmat(new_lambda_vec), mkIdxSet(ru),mkIdxSet(ou));

    // putting results into the class
    canonical_Lambda = new_lambda_vec;
    entanglement_spectrum = canonical_Lambda % canonical_Lambda;
    canonical_Gamma = new_Gamma;

    // checking canonicalization
    if (verbose)
        lfout << "right canonicalization check" << endl;

    UP_tensor = new_Gamma * new_lambda;
    UP_tensor.reIndex(ou,sul,sur,vu);
    DN_tensor = UP_tensor.conjugate();
    DN_tensor.reIndex(mkIdxSet(od,sul,sur,vd));
    Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
    arnoldi_res = arnoldi(Vec,UP_tensor,DN_tensor);
    if (verbose)
        lfout << arnoldi_res << endl;
    Vecmat = Vec.toMat(1,1);
    Vecmat = Vecmat /Vecmat.max(); // killing the irrelevant phase factor
    eig_sym(Vecvals,Vecmat);
    if (verbose)
        lfout << "right check is : " << endl << Vecvals;
    // TO-DO : add an assert for canonicalization check

    // left check
    /*
     * find the product of lambda from left to Gamma and find the new
     * left largest eigenvalue and the corresponding eigenvector
     */
    if (verbose)
        lfout << "left check" << endl;
    new_Gamma.reIndex(mkIdxSet(lu,sul,sur,ou));
    new_lambda.reIndex(mkIdxSet(vu,lu));
    // now UP_tensor must have Indexes: vu, sul, sur, ou
    UP_tensor = new_lambda * new_Gamma;
    DN_tensor = UP_tensor.conjugate();
    DN_tensor.reIndex(mkIdxSet(vd,sul,sur,od));
    Vec.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
    arnoldi_res = arnoldi(Vec,UP_tensor,DN_tensor);
    if (verbose)
        lfout << arnoldi_res << endl;
    //Vec.print(20);
    Vecmat = Vec.toMat(1,1);
    Vecmat = Vecmat/Vecmat.max(); // killing the irrelevant phase factor
    eig_sym(Vecvals,Vecmat);
    if (verbose)
        lfout << "left check is : " << endl << Vecvals;

    canonical_Gamma/sqrt(arnoldi_res.real());
    UP_tensor/sqrt(arnoldi_res.real());
    DN_tensor/sqrt(arnoldi_res.real());
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
    // TO-DO : improve Lanczos algorithm
    if (verbose)
        lfout << "starting lanczos!" << endl;
    double r_threshold = 1.0e-12;
    cx_vec r, trial, final;
    cx_mat Q;
    cx_vec q;
    mat T, eigenvecs;
    vec eigenvals;
    double error = 1.0;
    vector<double> alphas, betas;
    int i,step_counter = 0;
    r= guess.toVec();
    r = r/norm(r,2);
    trial = r; // for fidelity calculation
    double eigenresult;
    bool restart;
    while (error > 1.0e-15){
        restart = false;

        // first round (here r is normalized)
        if (abs(norm(r,2) - 1.0) < 1.0e-13 && verbose)
            lfout << norm(r,2) - 1.0 << endl;

        // clearing alphas and betas
        alphas.clear();
        betas.clear();

        assert(abs(norm(r,2) - 1.0) < 1.0e-10);
        q = r;
        r = operateH(q);
        alphas.push_back(cdot(q,r).real());

        r = r - (alphas[0] * q);
        betas.push_back(norm(r,2));

        // if r is already a eigenvector report that
        if (betas[0] < r_threshold){
            r = r + (alphas[0] * q);
            r = r / norm(r,2);
            eigenresult = alphas[0];
            break;
        }

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
            if (betas[i+1] < r_threshold)
                restart = true;
            Q = join_rows(Q,q);

            // expanding the T matrix
            T.resize(i+2,i+2);
            T(i+1,i+1) = alphas[i+1];
            T(i+1,i) = betas[i];
            T(i,i+1) = betas[i];

            // calculating the eigenvalues of T
            eig_sym(eigenvals, eigenvecs, T);

            // Error estimation and convergence test
            // beta * eigenvecs last row
            error = betas[i+1] * eigenvecs(i+1,0);
            if ( abs(error) < 1.0e-15 || restart)
                break;
            ++i;
        }
        step_counter +=i;
        r = Q*eigenvecs.col(0);
        eigenresult = eigenvals(0);
    }
    // fidelity calculations
    //final = Q*eigenvecs.col(0);
    final = r;
    //energy.push_back(eigenvals(i+1));
    energy.push_back(eigenresult);
    cx_d f = cdot(trial,final);
    if (verbose){
        lfout << "error: " << error
              << " , number of steps : " << i+1 << endl;
        lfout << "trial \\cdot final"<< f << endl;
    }
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
    // TO-DO : improve Arnoldi algorithm
    if (verbose)
        lfout << "starting arnoldi" << endl;
    double r_threshold = 1.0e-10;
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
    //bool restart;
    while (error > 1.0e-15){
        // operating UP DN V
        Vtemp.fromVec(Q.col(i),mkIdxSet(vu,vd));
        Vtemp = (Vtemp * UP_tensor) * DN_tensor;
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
        hbefore = norm(r,2);

        // eigensolving T
        eig_gen(eigenvals, eigenvecs, T);
        // eigenvals are not ordered
        // find the largest abs eigenvalue
        vec absvals = abs(eigenvals);
        absvals.max(sss);

        resV = Q * eigenvecs.col(sss);

        if (abs(hbefore) < r_threshold)
            break;

        // if (i > 100)
        //     break;

        // convergence
        errors = abs(eigenvecs.row(i)).st() * hbefore;
        error = errors(sss);
        q = r/hbefore;
        Q = join_rows(Q,q);

        i++;
    }
    if (verbose){
        lfout << "result is :" << eigenvals(sss) << endl;
        lfout << "finished in : " << i << "steps" << endl;
    }
    V.fromVec(resV, mkIdxSet(vu,vd));
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
        do_step();
        if (converged)
            break;
    }

    lfout << endl;

    int num_particles;

    if (verbose){
        // printing energy, Fidelity, truncations, D results at each level
        lfout << "|" <<  setw(15) << "ENERGY" << " |"
              << setw(16) << "Fidelity" << " |"
              << setw (16) << "Truncations" << " |" << setw(4) << "D"
              << " |" << setw(16) << "Convergence" << " |" << endl << endl;
        for (u_int i = 0 ; i < energy.size(); ++i) {
            num_particles = 2*(i+1);
            lfout << "|" << setw(15) << setprecision(8)
                  << energy[i]/num_particles + largestEV << " |"
                  << setw(16) << fidelity[i] << " |"
                  << setw(16) << lambda_truncated[i] << " |"
                  << setw (4) << matDims[i] << " |"
                  << setw (16) << convergence[i] << " |";
            lfout << endl;
        }
        lfout << endl;
        // printing canonically
        lfout << "canonical_Gammas" << endl;
        lfout << "canonical_Lambda:" << endl;
        lfout << entanglement_spectrum << endl;
    }

    // printing useful information
    lfout << "finished in " << iteration << " iteration" << endl;
    lfout << "final truncation error" << endl;
    lfout << lambda_truncated[energy.size()-1] << endl;
    lfout << "Von-Neumann : " << renyi(1.0,entanglement_spectrum) << endl;
    lfout << "Renyi  0.5  : " << renyi(0.5, entanglement_spectrum) << endl;
    lfout << "Renyi  2    : " << renyi(2.0, entanglement_spectrum) << endl;
    lfout << "Renyi  2    : " << renyi(100.0, entanglement_spectrum) << endl;
}

/**
 * expectation_onesite
 * calculates the expectation value of a given one-site operator
 * using canonical Lambda and Gammma
 * an example is S_z
 */
double IDMRG::expectation_onesite(cx_mat onesite_op){
    //double expectation_value = 0.0;
    u_int D = canonical_Lambda.size();
    Index lu = canonical_Gamma.indeces[0];
    Index ru = canonical_Gamma.indeces[3];
    Index il("il",D),ir("ir",D);
    cx_mat eyeD(D,D);
    eyeD.eye();
    lfout << "starting the one site calculation" << endl;
    // the one site operator is a d.d matrix
    Tensor onesite, lamleft, lamright;
    onesite.fromMat(kron(onesite_op, onesite_op) ,
                    mkIdxSet(sdl,sdr),mkIdxSet(sul,sur));
    lamleft.fromMat(eyeD * diagmat(canonical_Lambda),
                    mkIdxSet(il), mkIdxSet(lu));
    lamright.fromMat(eyeD * diagmat(canonical_Lambda),
                     mkIdxSet(ru), mkIdxSet(ir));

    Tensor Gamma_up = lamleft * canonical_Gamma * lamright;
    Tensor Gamma_dn = Gamma_up.conjugate();
    // Gamma_up.printIndeces();
    // Gamma_dn.reIndex(il,sdl,sdr,ir);
    // Gamma_dn.printIndeces();
    // onesite.printIndeces();
    Tensor result = Gamma_up * onesite * Gamma_dn;
    // result.printIndeces();
    // result.print(2);
    return result.values[0].real();
}

/**
 * expectation_twosite
 * calculates the expectation value of a given  two-site operator
 * using canonical Lambda and Gammma
 * an example is the energy for NN models
 */
double IDMRG::expectation_twosite(cx_mat twosite_op){
    double expectation_value = 0.0;
    Tensor Gamma_star = canonical_Gamma.conjugate();

    return expectation_value;
}

/**
 * gsFidelity
 * calculates the ground state fidelity
 * given the MPO for left and right Hamiltonians, which are different
 * for a small amount of change in the desired parameter
 */
double IDMRG::gsFidelity(cx_mat leftmatHamilt, cx_mat rightmatHamilt){
    double fid = 0.0;

    return fid;
}

/**
 * Renyi entroypy calculator
 */
double IDMRG::renyi(double alpha, const vec & L){

    double sumL = sum(L);
    // check for the sum of lambdas to be equal to 1
    if ( abs(sumL-1.0) > 1.0e-10 && verbose )
        lfout << "sum of Lambda squared is : " << sumL << endl;
    //assert ( abs(sumL-1.0) < 1.0e-10 );

    vec Ltemp;
    double result;
    // the resolution of this function is 10^-5
    if (abs(alpha - 1.0) < 1.0e-5)
        result = - sum ( L % log2(L) );
    else {
        Ltemp = pow(L,alpha);
        Ltemp = log2(Ltemp);
        result = - sum( Ltemp/ (1.0-alpha) );
    }
    return result;
}

Tensor IDMRG::get_Gamma() const{
    return canonical_Gamma;
}
vec IDMRG::get_Lambda() const{
    return canonical_Lambda;
}

double gsFidelity(const IDMRG & left, const IDMRG & right){
    // defining D
    u_int Dl = left.entanglement_spectrum.size();
    u_int Dr = right.entanglement_spectrum.size();
    u_int D = (Dl > Dr) ? Dr : Dl;
    cout << "Dl = " << Dl << ", Dr = " << Dr << " => D =" << D << endl;

    Index vu("vu", D), vd("vd", D), lu("lu", D), ru("ru", D), ld("ld", D), rd("rd",D);
    Tensor leftGam = left.get_Gamma();
    Tensor rightGam = right.get_Gamma().conjugate();
    Index sul = leftGam.indeces[1];
    Index sur = leftGam.indeces[2];
    leftGam = leftGam.slice(leftGam.indeces[0],0,D).slice(leftGam.indeces[3],0,D);
    rightGam = rightGam.slice(rightGam.indeces[0],0,D).slice(rightGam.indeces[3],0,D);
    Tensor leftLam, rightLam;
    cx_mat eyeD(D,D);
    eyeD.eye();
    leftLam.fromMat(eyeD * diagmat(left.get_Lambda()(span(0,D-1))), mkIdxSet(ru), mkIdxSet(vu));
    rightLam.fromMat(eyeD * diagmat(right.get_Lambda()(span(0,D-1))), mkIdxSet(ru), mkIdxSet(vd));
    cout << "constructing up and dn" << endl;
    Tensor up = leftGam * leftLam;
    Tensor dn = rightGam * rightLam;

    dn.reIndex(ld,sul,sur,vd);
    Tensor Vr;
    Vr.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
    cx_d c = arnoldi(Vr,up,dn);
    cout << "fidelity is " << c << endl;

    // leftGam.reIndex(lu,sul,sur,ru);
    // rightGam.reIndex(lu,sul,sur,rd);
    // leftLam.reIndex(mkIdxSet(vu,lu));
    // rightLam.reIndex(mkIdxSet(vd,lu));
    // up = leftLam * leftGam;
    // dn = rightLam * rightGam;

    // Tensor Vl;
    // Vl.fromVec(randu<cx_vec>(D*D),mkIdxSet(vu,vd));
    // cx_d b = arnoldi(Vl,up,dn);
    // cout << "fidelity is " << b << endl;

    // leftLam.reIndex(mkIdxSet(vu,lu));
    // rightLam.reIndex(mkIdxSet(vd,ld));
    // Vr.reIndex(mkIdxSet(lu,ld));

    // Tensor III = Vl * leftLam * rightLam * Vr;
    // III.print(1);
    // cx_d c = a * b;
    return c.real()*c.real() + c.imag()*c.imag();
}

cx_d arnoldi(Tensor & V, const Tensor & up, const Tensor & dn){
    double r_threshold = 1.0e-10;
    Index vu = V.indeces[0];
    Index vd = V.indeces[1];
    u_int D = vu.card;
    Tensor Vtemp;
    cx_vec h, resV;
    vec errors;
    cx_vec r = V.toVec();
    cx_vec q = r/norm(r,2);
    cx_mat T, Q = q;
    double error = 1;
    double hbefore = 0.0;
    cx_mat eigenvecs;
    cx_vec eigenvals;
    uword sss;
    //bool restart;
    for (int i = 0; i < D*D; ++i){
        // operating UP DN V
        Vtemp.fromVec(Q.col(i),mkIdxSet(vu,vd));
        Vtemp = (Vtemp * up) * dn;
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
        hbefore = norm(r,2);

        // eigensolving T
        eig_gen(eigenvals, eigenvecs, T);
        // eigenvals are not ordered
        // find the largest abs eigenvalue
        vec absvals = abs(eigenvals);
        absvals.max(sss);

        resV = Q * eigenvecs.col(sss);

        if (abs(hbefore) < r_threshold)
            break;


        // convergence
        errors = abs(eigenvecs.row(i)).st() * hbefore;
        error = errors(sss);
        q = r/hbefore;
        Q = join_rows(Q,q);

        if (error < 1.e-14)
            break;

    }
    V.fromVec(resV, mkIdxSet(vu,vd));
    return eigenvals(sss);
}
