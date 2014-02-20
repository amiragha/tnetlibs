#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include "iDMRG.h"

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
    finalD = 0;
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

    //arnoldi = new Arnoldi(&applyUPDN);
    /* Note:
     * here WL has Indeces = sdl:d, sul:d, b:B
     * here WR has Indeces = sdr:d, b:B, sur:d
     * then W  has Indeces = sdl:d, sul:d, sdr:d, sur:d
     */
    vec eigenvals;
    cx_mat eyed(d,d);
    eyed.eye();
    WL.fromMat(matHamilt_lmost, mkIdxSet(sdl), mkIdxSet(sul,b));
    WR.fromMat(matHamilt_rmost, mkIdxSet(sdr,b), mkIdxSet(sur));
    W = WL * WR;
    W.rearrange(mkIdxSet(sdl,sdr,sul,sur));
    energyMPO = W.toMat(2,2);
    eig_sym(eigenvals,energyMPO);
    largestEV = eigenvals(d*d - 1);
    matHamilt.submat(B*d-d,0,B*d-1,d-1) =
        matHamilt.submat(B*d-d,0,B*d-1,d-1) -largestEV * eyed;
    matHamilt_lmost = matHamilt.rows(B*d-d, B*d-1);
    matHamilt_rmost = matHamilt.cols(0,d-1);

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

IDMRG::IDMRG(arma::cx_mat& mHamilt, u_int Bdim, u_int dim, u_int mD,
             Tensor& in_left, Tensor& in_right,
             arma::cx_vec& in_guess, arma::vec& in_llamb,
             double con_thresh, bool in_verbose,
             std::string logfile)
{
    verbose = in_verbose;
    B = Bdim;
    d = dim;
    maxD = mD;
    finalD = 0;
    converged = false;
    convergence_threshold = con_thresh;

    Left  = in_left;
    Right = in_right;
    guess = in_guess;
    llamb = in_llamb;
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
    u_int initialD = llamb.size();
    matDims.push_back(initialD);

    // initialization of needed indeces for the whole class
    // TO-DO : find a better initialization for indeces
    bl.i("bl",B), br.i("br",B), b.i("b",B),
        sul.i("sul",d), sdl.i("sdl",d), sur.i("sur",d), sdr.i("sdr",d),
        sd.i("sd",d), su.i("su",d);

    //arnoldi = new Arnoldi(&applyUPDN);
    /* Note:
     * here WL has Indeces = sdl:d, sul:d, b:B
     * here WR has Indeces = sdr:d, b:B, sur:d
     * then W  has Indeces = sdl:d, sul:d, sdr:d, sur:d
     */
    vec eigenvals;
    cx_mat eyed(d,d);
    eyed.eye();
    WL.fromMat(matHamilt_lmost, mkIdxSet(sdl), mkIdxSet(sul,b));
    WR.fromMat(matHamilt_rmost, mkIdxSet(sdr,b), mkIdxSet(sur));
    W = WL * WR;
    W.rearrange(mkIdxSet(sdl,sdr,sul,sur));
    energyMPO = W.toMat(2,2);
    eig_sym(eigenvals,energyMPO);
    largestEV = eigenvals(d*d - 1);
    matHamilt.submat(B*d-d,0,B*d-1,d-1) =
        matHamilt.submat(B*d-d,0,B*d-1,d-1) -largestEV * eyed;
    matHamilt_lmost = matHamilt.rows(B*d-d, B*d-1);
    matHamilt_rmost = matHamilt.cols(0,d-1);

    WL.fromMat(matHamilt_lmost, mkIdxSet(sdl), mkIdxSet(sul,b));
    WR.fromMat(matHamilt_rmost, mkIdxSet(sdr,b), mkIdxSet(sur));
    W = WL * WR;
    W.rearrange(mkIdxSet(sdl,sdr,sul,sur));

    // defining Hamiltonian Tensors
    // constructing Hamiltonian Tensors from matrices
    Hamilt.fromMat(matHamilt, mkIdxSet(sd,bl), mkIdxSet(su,br));

    WL = Hamilt;
    WR = Hamilt;
    WL.reIndex(mkIdxSet(sdl,bl,sul,b));
    WR.reIndex(mkIdxSet(sdr,b,sur,br));

    // W indeces : sdl:d, bl:B, sul:d, sdr:d, sur:d, br:B
    W = WL * WR;
    WL.reIndex(mkIdxSet(sdl,b,sul,bl));
    WR.reIndex(mkIdxSet(sdr,br,sur,b));

    iteration = 0;


    lfout.open(logfile.c_str());
    // start the initial setup
    zeroth_iter_with_init();
    iterate();
    lfout.close();
}

IDMRG::~IDMRG(){
    //delete arnoldi;
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
IDMRG::lambda_size_trunc (const vec & S)
{
    double threshold = 1.0e-15;
    u_int nextD = 0;
    for (nextD = 0; nextD < S.size(); ++nextD)
    {
        if (abs(S(nextD) < threshold)) break;
    }
    // nextD is now on it's proper value ( if not larger than maxD)
    if (nextD > maxD) nextD = maxD;

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
IDMRG::zeroth_iter()
{
    if (verbose)
        lfout << "Zeroth level starting" << endl;

    int nextD;

    // solving the eigenvalue problem for W.toMat(2,2);
    cx_mat eigenvecs;
    vec eigenvals;
    eig_sym(eigenvals,eigenvecs, W.toMat(2,2));

    /*
     * Note :
     * here eigenvecs.col(0) is d.d vector
     * we have to reshape it into a d*d matrix
     */
    cx_mat U,V;
    vec S;
    svd(U,S,V,reshape(eigenvecs.col(0),d,d),"std");

    // the starting energy for the 2site problem
    energy.push_back(eigenvals(0));
    guessFidelity.push_back(1);
    convergence.push_back(1);

    // lambda size check
    nextD = lambda_size_trunc(S);

    // the truncation step
    cx_mat U_trunc = U.cols(0, nextD-1);
    cx_mat V_trunc = V.cols(0, nextD-1);
    vec S_trunc = S(span(0,nextD-1));

    // stroing the lambda
    // truncated lambda
    double lambda_norm = norm(S_trunc,2);
    truncations.push_back(1-lambda_norm);
    lambda.push_back(S_trunc/lambda_norm);

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
    // cx_mat guessmat = randu<cx_mat>(d*nextD,d*nextD);
    // guess.fromMat(guessmat, mkIdxSet(sul,lu), mkIdxSet(sur,ru));
    guess = randu<cx_vec>(d*nextD*d*nextD);
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

void
IDMRG::zeroth_iter_with_init()
{

    cout << "zero" << endl;
    iteration = 0;

    u_int nextD, D = matDims.back();

    // needed indeces
    Index lu("lu", D), ru("ru", D),ld("ld", D), rd("rd", D);

    // lanczos step
    cx_vec ksiVec;
    energy.push_back(Lanczos(guess, ksiVec));

    // guess fidelity calculations
    guessFidelity.push_back(1.0 - abs(cdot(guess/norm(guess,2), ksiVec)));
    cx_mat U,V;
    vec S;
    //svd(U,S,V,ksi.toMat( mkIdxSet(lu,sul), mkIdxSet(ru,sur) ) ,"std);
    svd_econ(U,S,V,reshape(ksiVec,D*d,D*d),'b', "std");

    // lambda size check
    nextD = lambda_size_trunc(S);

    // truncation step
    cx_mat U_trunc = U.cols(0, nextD-1);
    cx_mat V_trunc = V.cols(0, nextD-1);
    vec S_trunc = S(span(0, nextD-1));

    // truncated lambda
    double lambda_norm = norm(S_trunc,2);
    truncations.push_back(1-lambda_norm);

    // normalizing lambda
    lambda.push_back(S_trunc/lambda_norm);

    mat S_trunc_mat = diagmat(S_trunc/lambda_norm);

    // calculating guess for the next iteration
    convergence.push_back(1);

    cx_mat newA, newB, left_lambda, right_lambda,u,v;
    //mat diags;
    vec s;
    cx_mat AL  = U_trunc * S_trunc_mat;
    cx_mat lft;
    lft.set_size(D,nextD*d);
    for (u_int i = 0; i<d; ++i)
    {
        lft.submat(0,i*nextD,D-1,(i+1)*nextD-1) =
            AL.submat(i*D,0,(i+1)*D-1,nextD-1);
    }

    svd_econ(u, s, newB, lft,'b',"std");
    newB = newB.t();
    left_lambda = u*diagmat(s);
    cx_mat LB  = S_trunc_mat * V_trunc.t();
    cx_mat rgt;
    rgt.set_size(nextD*d,D);
    for (u_int i = 0; i<d; ++i)
    {
        rgt.submat(i*nextD,0,(i+1)*nextD-1,D-1) =
            LB.submat(0,i*D,nextD-1,(i+1)*D-1);
    }

    svd_econ(newA,s,v,rgt,'b',"std");
    right_lambda = diagmat(s)*v.t();
    mat guesscore = inv(diagmat(llamb));

    cx_mat guessMat = newA * right_lambda * guesscore * left_lambda * newB;
    // if a fully random is needed
    //cx_mat guessMat = randu<cx_mat>(d*nextD,d*nextD);

    // building the Tensor guess
    guess = reshape(guessMat,d*d*nextD*nextD,1);

    update_LR(U_trunc, V_trunc, D, nextD);

}

/**
 * do_step
 * go one step forward in the iDMRG algorithm
 */
void
IDMRG::do_step()
{
    iteration++;

    if (verbose)
        lfout << "Iteration No. " << iteration << std::endl;

    u_int nextD, D = matDims.back();

    // needed indeces
    Index lu("lu", D), ru("ru", D),ld("ld", D), rd("rd", D);

    // lanczos step
    cx_vec ksiVec;
    energy.push_back(Lanczos(guess, ksiVec));

    // exact computation instead of lanczos
    // Left.reIndex(mkIdxSet(lu,bl,ld));
    // Right.reIndex(mkIdxSet(ru,br,rd));

    // Tensor HH;
    // HH = (Left * W) * Right;
    // HH.rearrange(mkIdxSet(sdl,ld,sdr,rd,sul,lu,sur,ru));
    // cx_mat eigenVectors;
    // vec eigenValues;
    // eig_sym(eigenValues, eigenVectors, HH.toMat(4,4));
    // energy.push_back(eigenValues(0));
    // ksiVec = eigenVectors.col(0);


    // guess fidelity calculations
    guessFidelity.push_back(1.0 - abs(cdot(guess/norm(guess,2), ksiVec)));


    /*
     * Note:
     * here we received the ksiVec from lanczos
     * we suppose that the result of the lanczos vector has the following
     * indeces = sul:d ,lu:D ,sur:d ,ru;D
     * correction =
     * indeces = lu:D, sul:d ,ru;D, sur:d
     */

    // vector<Index> ksiInd = mkIdxSet(sul,lu,sur,ru);
    // Tensor ksi(ksiInd);
    // ksi.fromVec(ksiVec, mkIdxSet(sul,lu,sur,ru));

    cx_mat U,V;
    vec S;
    //svd(U,S,V,ksi.toMat( mkIdxSet(lu,sul), mkIdxSet(ru,sur) ) ,"std);
    svd_econ(U,S,V,reshape(ksiVec,D*d,D*d),'b',"std");

    // lambda size check
    nextD = lambda_size_trunc(S);

    // truncation step
    cx_mat U_trunc = U.cols(0, nextD-1);
    cx_mat V_trunc = V.cols(0, nextD-1);
    vec S_trunc = S(span(0, nextD-1));

    // truncated lambda
    double lambda_norm = norm(S_trunc,2);
    truncations.push_back(1-lambda_norm);

    // normalizing lambda
    lambda.push_back(S_trunc/lambda_norm);

    mat S_trunc_mat = diagmat(S_trunc/lambda_norm);

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
    //mat diags;
    vec s;
    /*
     * rotate left
     * note: U_trunc * S_trunc is a matrix of D.d*nextD dimension
     * we need to perform svd on the matrix D*d.nex,"std"tD
     *
     * and since only we have QR factorization we use it on lft.t() and
     * then find the Q.t() and R.t() to perform a RQ transition
     */

    cx_mat AL  = U * S;
    cx_mat lft;
    lft.set_size(D,nextD*d);
    for (u_int i = 0; i<d; ++i)
    {
        lft.submat(0,i*nextD,D-1,(i+1)*nextD-1) =
            AL.submat(i*D,0,(i+1)*D-1,nextD-1);
    }

    svd_econ(u, s, newB, lft,'b',"std");
    newB = newB.t();
    left_lambda = u*diagmat(s);

    /*
     * rotate right
     * note: S_trunc * V.t()_trunc is a matrix of nextD*D.d dimension
     * we need to perform svd on the matrix d.nextd,"std"*D
     */

    cx_mat LB  = S * V.t();
    cx_mat rgt;
    rgt.set_size(nextD*d,D);
    for (u_int i = 0; i<d; ++i)
    {
        rgt.submat(i*nextD,0,(i+1)*nextD-1,D-1) =
            LB.submat(0,i*D,nextD-1,(i+1)*D-1);
    }

    svd_econ(newA,s,v,rgt,'b',"std");
    right_lambda = diagmat(s)*v.t();


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

    // building the Tensor guess
    guess = reshape(guessMat,d*d*nextD*nextD,1);

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

    if (!converged && iteration < 300){
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
void IDMRG::canonicalize(Tensor A, Tensor B, u_int D, u_int nextD)
{
    // canonicalization
    if (verbose)
        lfout << "starting Canonilcalization" << endl;

    u_int n = lambda.size()-1;

    // needed indeces
    Index plu = A.indeces[0];
    Index nlu = A.indeces[2];
    Index nru = B.indeces[0];
    Index pru = B.indeces[1];

    /*
     * Justify truncation:
     * in the process of canonicalization D and nextD must be equal, so
     * if one is larger then truncation is necessary. In addition, if some
     * of the eigen values of right and left eigenvectors of the transfer
     * matrix is zero then we have to further reduce the dimension of the
     * matrices.
     */

    // the process of canonicalization has 5 steps.
    // 1) re-indexing tensors
    // 2) UP, DN Tensor definitions
    // 3) sending to Arnoldi
    // 4) eigen solving the left/right eigen vector
    // 5) naming the eigen decomposition of left/right eigen vector

    // needed Indexes and Tensors
    Index gmr("gmr", D), gml("gml", D), ou("ou", D), od("od", D),
        vu("vu", D), vd("vd", D);
    Tensor inv_lambda_B, lambda_A;
    cx_mat Vecmat, lft_vecs, rgt_vecs, X, Y;
    vec lft_vals, rgt_vals;
    cx_d arnoldi_res;
    cx_vec invec;

    vec lambda_B_vec = lambda[n-1];
    cx_mat lambda_B_mat;

    lambda_A.fromMat(eye<cx_mat>(nextD, nextD) * diagmat(lambda[n]),
                     mkIdxSet(gml),mkIdxSet(gmr));

    u_int currentD;
    finalD = D;
    while (true)
    {
        ou.card = finalD;
        od.card = finalD;
        vu.card = finalD;
        vd.card = finalD;
        lambda_B_mat = eye<cx_mat>(finalD, finalD) * diagmat(lambda_B_vec);
        inv_lambda_B.fromMat(inv(lambda_B_mat),
                             mkIdxSet(pru), mkIdxSet(ou));
        // left canonicalization
        // 1)
        A.reIndex(mkIdxSet(vu,sul,gml));
        B.reIndex(mkIdxSet(gmr,pru,sur));
        // 2)
        UP_tensor = A * lambda_A * B * inv_lambda_B;
        DN_tensor = UP_tensor.conjugate();
        DN_tensor.reIndex(mkIdxSet(vd,sul,sur,od));
        // 3)
        invec = randu<cx_vec>(finalD*finalD);
        arnoldi_res = arnoldi(invec, Vecmat);
        if (verbose) lfout << arnoldi_res << endl;
        Vecmat /= Vecmat.max();
        Vecmat /= norm(Vecmat,2);
        // 4)
        eig_sym(lft_vals,lft_vecs,reshape(Vecmat, finalD, finalD));
        if (verbose) lfout << "left : " << endl << lft_vals << endl;

        // right canonicalization
        // 1)
        inv_lambda_B.reIndex(mkIdxSet(ou,plu));
        A.reIndex(mkIdxSet(plu, sul, gml));
        B.reIndex(mkIdxSet(gmr, vu, sur));
        // 2)
        UP_tensor = inv_lambda_B * A * lambda_A * B;
        DN_tensor = UP_tensor.conjugate();
        DN_tensor.reIndex(mkIdxSet(od,sul,vd,sur));
        // 3)
        invec = randu<cx_vec>(finalD*finalD);
        arnoldi_res = arnoldi(invec, Vecmat);
        if (verbose) lfout << arnoldi_res << endl;
        Vecmat /= Vecmat.max();
        Vecmat /= norm(Vecmat,2);
        // 4)
        eig_sym(rgt_vals,rgt_vecs,reshape(Vecmat, finalD, finalD));
        if (verbose) lfout << "right : " << endl << rgt_vals << endl;

        // remove the zeros
        for (currentD = 0; currentD < lft_vals.size(); ++currentD)
        {
            if (abs(lft_vals[currentD]) > 1.0e-14 &&
                abs(rgt_vals[currentD]) > 1.0e-14) break;
        }
        currentD = finalD - currentD;

        if (finalD == currentD) break;
        finalD = currentD;
        cout << "finalD = " << finalD << endl;

        A.reIndex(mkIdxSet(plu, sul, nlu));
        B.reIndex(mkIdxSet(nru, pru, sur));

        A = A.slice(plu,0,finalD-1);
        plu.card = finalD;

        B = B.slice(pru,0,finalD-1);
        pru.card = finalD;
        // truncate lambda_B
        lambda_B_vec = lambda_B_vec(span(0,finalD-1));
        lambda_B_vec /= norm(lambda_B_vec, 2);
    }

    // 5)
    Y = diagmat(sqrt(lft_vals)) * lft_vecs.st();
    X = rgt_vecs * diagmat(sqrt(rgt_vals));

    if (verbose){
        lfout << "old Von-Neuman entropy is : " << renyi(1.0,lambda_B_vec)<< endl;
        lfout << "defining new lambda and Gamma" << endl;
    }


    // defining new lambda and new gamma
    Index tl("tl", finalD), tr("tr",finalD), lu("lu",finalD), ru("ru", finalD);
    cx_mat U,V;

    svd(U, canonical_Lambda, V, (Y * lambda_B_mat * X) ,"std");
    canonical_Lambda /= norm(canonical_Lambda,2);

    cx_mat templeft_mat, tempright_mat;
    Tensor templeft, tempright, rLambar, lLambar, new_lambda;
    rLambar = inv_lambda_B;
    lLambar = inv_lambda_B;
    rLambar.reIndex(mkIdxSet(pru, tr));
    lLambar.reIndex(mkIdxSet(tl, plu));
    A.reIndex(mkIdxSet(plu, sul, gml));
    B.reIndex(mkIdxSet(gmr, pru, sur));

    templeft_mat = V.t() * inv(X);
    templeft.fromMat(templeft_mat,mkIdxSet(lu),mkIdxSet(tl));
    tempright_mat = inv(Y) * U;
    tempright.fromMat(tempright_mat,mkIdxSet(tr),mkIdxSet(ru));
    canonical_Gamma =
        templeft *  lLambar * A * lambda_A * B * rLambar * tempright;

    // entanglement spectrum
    entanglement_spectrum = canonical_Lambda % canonical_Lambda;

    // checking canonicalization
    if (verbose)
        lfout << "right canonicalization check" << endl;

    // 1),2)
    //UP_tensor = new_Gamma * new_lambda;
    UP_tensor = get_GL();
    UP_tensor.reIndex(ou,sul,sur,vu);
    DN_tensor = UP_tensor.conjugate();
    DN_tensor.reIndex(mkIdxSet(od,sul,sur,vd));
    // 3)
    invec = randu<cx_vec>(finalD*finalD);
    arnoldi_res = arnoldi(invec, Vecmat,true);
    if (verbose) lfout << arnoldi_res << endl;
    Vecmat /= Vecmat(0);
    // 4)
    eig_sym(rgt_vals,rgt_vecs,reshape(Vecmat, finalD, finalD));
    if (verbose) lfout << "right check is: " << endl << rgt_vals << endl;

    // left check
    if (verbose)
        lfout << "left check" << endl;
    // 1),2)
    UP_tensor = get_LG();
    UP_tensor.reIndex(mkIdxSet(vu,sul,sur,ou));
    DN_tensor = UP_tensor.conjugate();
    DN_tensor.reIndex(mkIdxSet(vd,sul,sur,od));
    // 3)
    invec = randu<cx_vec>(finalD*finalD);
    arnoldi_res = arnoldi(invec, Vecmat,true);
    if (verbose) lfout << arnoldi_res << endl;
    Vecmat /= Vecmat(0);
    // 4)
    eig_sym(lft_vals,rgt_vecs,reshape(Vecmat, finalD, finalD));
    if (verbose) lfout << "left check is: " << endl << lft_vals << endl;

    // normalizing the canonical_Gamma
    canonical_Gamma/sqrt(arnoldi_res.real());

}

/**
 * operateH
 * find the effect of bigH(Hamiltonians and Left and Right) on a given
 * vector
 * converts it to Tensor first, performs the contraction and then gives the
 * result back as vector
 * input of operateH must be a vector built out of a tensor
 * with indeces = lu sul ru sur
 */
void
IDMRG::operateH(cx_vec & q, cx_vec & res){
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
    // Left.printIndeces();
    // Right.printIndeces();
    // W.printIndeces();

    // given Tensor must have the indeces : sul, lu, sur, ru
    // Tensor given;
    // given.fromVec(q,mkIdxSet(sul, lu, sur, ru));

    Tensor given;
    given.fromVec(q,mkIdxSet(lu, sul, ru, sur));

    Tensor result;
    result = ((Left * given) * W) * Right;

    /*
     * after this operation the restult have : ld sdl sdr rd
     * so a rearrangement is necessary so the result will be a vector
     * built from a tensor with indeces = sdl ld sdr rd
     */
    //result.rearrange(mkIdxSet(sdl, ld, sdr, rd));
    result.rearrange(mkIdxSet(ld, sdl, rd, sdr));

    res = result.toVec();
}

/**
 * applyUPDN
 */
void IDMRG::applyUPDN(const cx_vec & in, cx_vec & out)
{
    u_int D = UP_tensor.indeces[0].card;
    assert (D*D == in.size());
    Index vu("vu",D),vd("vd",D);
    Tensor V;
    V.fromVec(in, mkIdxSet(vu,vd));
    V = (V * UP_tensor) * DN_tensor;
    out = V.toVec();
}
/**
 * Lanczos
 * given a guess vector and a reference ksi, updates the ksi to the
 * eigenvector with smallest eigenvalue of L*W*R
 * input guess Tensor and the ksi vector as reference
 *
 * return void
 */
double
IDMRG::Lanczos(arma::cx_vec& vstart, arma::cx_vec& eigenVector)
{
    // TO-DO : improve Lanczos algorithm
    if (verbose)
        lfout << "starting lanczos!" << endl;

    u_int                maxLaSpace = vstart.size();
    // defining needed variables
    arma::cx_vec         q, r;
    arma::cx_mat         Q;
    arma::mat            T, eigenVecsT;
    arma::vec            eigenValsT;
    arma::uvec           sorted_indeces;
    arma::uword          sorted_index;
    double               eigenValue;
    double               error;
    double               alpha, alphac,  beta = 0.0;
    bool                 all_converged = false;
    double               conv_thresh = 1.e-14;

    q = vstart / arma::norm(vstart,2);
    Q = q;

    u_int i;
    for (i = 0; i < maxLaSpace; ++i)
    {
        // build r , r = A * q
        operateH(q, r);

        // calculation of alpha:
        // although using the abs() function here seems plausible as well
        // using only the real part of the product is more numerically stable.
        alpha = cdot(q,r).real();

        // orthogonalization step
        r = r - alpha * q;

        alphac = cdot(q,r).real();
        r = r - alphac * q;

        alpha = alpha + alphac;
        r = r - Q * (Q.t() * r);

        // creating the matrix T
        if (i == 0) T << alpha;
        else
        {
            T.resize(i+1,i+1);
            T.col(i) = arma::zeros<colvec>(i+1);
            T.row(i) = arma::zeros<rowvec>(i+1);
            T(i,i) = alpha;
            T(i,i-1) = beta;
            T(i-1,i) = beta;
        }
        beta = arma::norm(r,2);

        // solve Aegean problem for T
        arma::eig_sym(eigenValsT, eigenVecsT, T);

        // eigenvals are ordered
        // find the largest eigenvalue (according to abs)
        sorted_indeces = arma::sort_index(abs(eigenValsT),1);
        arma::vec errs = abs(eigenVecsT.row(i)).st() * beta;

        all_converged = true;
        sorted_index = sorted_indeces(0);
        error = errs(sorted_index);
        if (error > 1.0e-14)
            all_converged = false;


        if (beta < conv_thresh && -beta < conv_thresh) break;
        if (error > 1.0e-14) all_converged = false;
        if (all_converged) break;
        if (i+1 == maxLaSpace) break;

        q = r/beta;
        Q = join_rows(Q,q);

    }

    sorted_index = sorted_indeces(0);
    eigenValue = eigenValsT(0);
    eigenVector = Q * eigenVecsT.col(0);

    //double r_threshold = 1.0e-12;
    //r= guess.toVec();

    return eigenValue;
}


/**
 * arnoldi_canonical
 * performs arnoldi algorithms using UP_tensor and DN_tensor
 * a part of canonicalization process
 *
 */
cx_d IDMRG::arnoldi_canonical(Tensor & V)
{
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
            T.row(i) = arma::zeros<cx_rowvec>(i+1);
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
    int N = 300;
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
              << setw(16) << "guessFidel" << " |"
              << setw (16) << "Truncations" << " |" << setw(4) << "D"
              << " |" << setw(16) << "Convergence" << " |" << endl << endl;
        for (u_int i = 0 ; i < energy.size(); ++i) {
            num_particles = 2*(i+1);
            lfout << "|" << setw(15) << setprecision(8)
                  << energy[i]/num_particles + largestEV << " |"
                  << setw(16) << guessFidelity[i] << " |"
                  << setw(16) << truncations[i] << " |"
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

    // final energy calcualtions
    mFinalEnergy = expectation_twosite(energyMPO);

    // printing useful information
    lfout << "finished in " << iteration << " iteration" << endl;
    lfout << "final truncation error : "
          << truncations[energy.size()-1] << endl;
    lfout << "final energy : " << mFinalEnergy << endl;
    lfout << "Von-Neumann : " << renyi(1.0,entanglement_spectrum) << endl;
    lfout << "Renyi  0.5  : " << renyi(0.5, entanglement_spectrum) << endl;
    lfout << "Renyi  2    : " << renyi(2.0, entanglement_spectrum) << endl;
    lfout << "Renyi  100  : " << renyi(100.0, entanglement_spectrum) << endl;
}

/**
 * expectation_onesite
 * calculates the expectation value of a given one-site operator
 * using canonical Lambda and Gammma
 * an example is S_z
 */
double IDMRG::expectation_onesite(cx_mat onesite_op){
    u_int D = canonical_Lambda.size();
    Index lu = canonical_Gamma.indeces[0];
    Index ru = canonical_Gamma.indeces[3];
    Index il("il",D),ir("ir",D);
    cx_mat eyeD(D,D);
    eyeD.eye();
    // the one site operator is a d.d matrix
    Tensor onesite, lamleft, lamright;
    onesite.fromMat(kron(onesite_op, eye(d,d)) + kron(eye(d,d), onesite_op),
                    mkIdxSet(sdl,sdr),mkIdxSet(sul,sur));
    lamleft.fromMat(eyeD * diagmat(canonical_Lambda),
                    mkIdxSet(il), mkIdxSet(lu));
    lamright.fromMat(eyeD * diagmat(canonical_Lambda),
                     mkIdxSet(ru), mkIdxSet(ir));

    Tensor Gamma_up = lamleft * canonical_Gamma * lamright;
    Tensor Gamma_dn = Gamma_up.conjugate();
    // Gamma_up.printIndeces();
    Gamma_dn.reIndex(il,sdl,sdr,ir);
    // Gamma_dn.printIndeces();
    // onesite.printIndeces();
    Tensor result = Gamma_up * onesite * Gamma_dn;
    // result.printIndeces();
    // result.print(2);
    return result.values[0].real()/2.0;
}

/**
 * expectation_twosite
 * calculates the expectation value of a given  two-site operator
 * using canonical Lambda and Gammma
 * an example is the energy for NN models
 */
double IDMRG::expectation_twosite(cx_mat twosite_op){
    u_int D = canonical_Lambda.size();
    Index lu = canonical_Gamma.indeces[0];
    Index ru = canonical_Gamma.indeces[3];
    Index il("il",D), ir("ir",D), lu2("lu2",D), ru2("ru2",D);
    Index sul2("sul2", d), sur2("sur2", d), sdl2("sdl2", d);
    cx_mat eyeD(D,D);
    eyeD.eye();
    // the one site operator is a d.d matrix
    Tensor canonical_Gamma2;
    canonical_Gamma2 = canonical_Gamma;
    canonical_Gamma2.reIndex(lu2,sul2,sur2,ru2);

    Tensor twosite, lamleft, lambetw, lamright;
    twosite.fromMat(kron(twosite_op, eye(d,d)) + kron(eye(d,d), twosite_op),
                    mkIdxSet(sdl,sdr,sdl2),mkIdxSet(sul,sur,sul2));
    lamleft.fromMat(eyeD * diagmat(canonical_Lambda),
                    mkIdxSet(il), mkIdxSet(lu));
    lambetw.fromMat(eyeD * diagmat(canonical_Lambda),
                    mkIdxSet(ru), mkIdxSet(lu2));
    lamright.fromMat(eyeD * diagmat(canonical_Lambda),
                     mkIdxSet(ru2), mkIdxSet(ir));

    Tensor up =
        lamleft * canonical_Gamma * lambetw * canonical_Gamma2 * lamright;
    Tensor dn = up.conjugate();
    // Gamma_up.printIndeces();
    dn.reIndex(mkIdxSet(il,sdl,sdr,sdl2, sur2,ir));
    // Gamma_dn.printIndeces();
    // onesite.printIndeces();
    Tensor result = up * twosite * dn;

    return result.values[0].real()/2.0;
}

double IDMRG::SymmetryEffect(cx_mat symmetry_op){
    // if (verbose) lfout << "testing symmetry effect" << endl;
    // UP_tensor = get_GL();
    // UP_tensor.reIndex(ou,sul,sur,vu);
    // DN_tensor = UP_tensor.conjugate();
    // DN_tensor.reIndex(mkIdxSet(od,sul,sur,vd));

    // arma::cx_vec invec = randu<cx_vec>(finalD*finalD);
    // double arnoldi_res = arnoldi(invec, Vecmat);
    // if (verbose) lfout << arnoldi_res << endl;

    return 0.9;
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

Tensor IDMRG::get_GL() const
{
    Index ru("ru", finalD), lu("lu", finalD), rru("rru", finalD);
    Tensor c_lambda;
    c_lambda.fromMat(eye<cx_mat>(finalD, finalD) * diagmat(canonical_Lambda)
                     , mkIdxSet(ru), mkIdxSet(rru));
    Tensor result = canonical_Gamma * c_lambda;
    result.reIndex(lu, sul, sur, ru);
    return result;
}

Tensor IDMRG::get_LG() const
{
    Index ru("ru", finalD), lu("lu", finalD), llu("llu", finalD);
    Tensor c_lambda;
    c_lambda.fromMat(eye<cx_mat>(finalD, finalD) * diagmat(canonical_Lambda)
                     , mkIdxSet(llu), mkIdxSet(lu));
    Tensor result = c_lambda * canonical_Gamma;
    result.reIndex(lu, sul, sur, ru);
    return result;
}

Tensor IDMRG::get_LGL() const
{
    Index ru("ru", finalD), lu("lu", finalD),
        llu("llu", finalD), rru("rru", finalD);
    Tensor c_lambda_left, c_lambda_right;
    c_lambda_left.fromMat(eye<cx_mat>(finalD, finalD)
                          * diagmat(canonical_Lambda)
                          , mkIdxSet(llu), mkIdxSet(lu));
    c_lambda_right.fromMat(eye<cx_mat>(finalD, finalD)
                           * diagmat(canonical_Lambda)
                           , mkIdxSet(ru), mkIdxSet(rru));
    Tensor result = c_lambda_left * canonical_Gamma * c_lambda_right;
    result.reIndex(lu, sul, sur, ru);
    return result;
}

Tensor IDMRG::get_Left() const
{
    return Left;
}
Tensor IDMRG::get_Right() const
{
    return Right;
}
arma::cx_vec IDMRG::get_guess() const
{
    return guess;
}
arma::vec IDMRG::get_llamb() const
{
    return lambda.back();
}

Tensor IDMRG::get_Gamma() const{
    return canonical_Gamma;
}
vec IDMRG::get_Lambda() const{
    return canonical_Lambda;
}

vec gsFidelity(const IDMRG & left, const IDMRG & right){
    // defining D
    u_int Dl = left.entanglement_spectrum.size();
    u_int Dr = right.entanglement_spectrum.size();
    u_int D = (Dl > Dr) ? Dr : Dl;
    cout << "Dl = " << Dl << ", Dr = " << Dr << " => D =" << D << endl;

    Index vu("vu", D), vd("vd", D),
        lu("lu", D), ru("ru", D), ld("ld", D), rd("rd",D);

    Tensor up = left.get_GL();
    Tensor dn = right.get_GL();
    dn = dn.conjugate();
    Index sul = up.indeces[1];
    Index sur = up.indeces[2];
    // asserting the equality of sur, sul in dn/up tensors
    assert (sul == dn.indeces[1]);
    assert (sur == dn.indeces[2]);

    // needed truncation
    up = up.slice(up.indeces[0],0,D).slice(up.indeces[3],0,D);
    dn = dn.slice(dn.indeces[0],0,D).slice(dn.indeces[3],0,D);

    up.reIndex(lu, sul, sur, ru);
    dn.reIndex(ld, sul, sur, rd);

    Tensor Vr = up * dn;
    Vr.printIndeces();
    Vr.rearrange(mkIdxSet(ld,lu,rd,ru));
    cx_vec tranferEig;
    cx_mat dummy;
    eig_gen(tranferEig, dummy, Vr.toMat(2,2));

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
    return abs(tranferEig);
}
cx_d IDMRG::arnoldi(arma::cx_vec&  vstart,
                    arma::cx_mat& eigenVectors,
                    bool correlation_calculation
                    )
{
    double conv_thresh = 1.e-15;
    arma::cx_vec eigenValue(1);
    u_int maxArSpace = vstart.size();
    u_int number_of_eigs = 1;

    // defining needed variables
    arma::cx_vec         q, r, h, c,eigenValsT;
    arma::cx_mat         Q, eigenVecsT;
    arma::cx_mat         T;
    arma::vec            errors;
    arma::uvec           sorted_indeces;
    arma::uword          sorted_index;
    double               hbefore = 0.0;
    bool                 all_converged = false;
    u_int                vectorDim = vstart.size();

    // initialize
    eigenVectors = arma::zeros<arma::cx_mat>(vectorDim, number_of_eigs);
    eigenValue   = arma::zeros<arma::cx_vec>(number_of_eigs);
    errors       = arma::ones<arma::vec>(number_of_eigs);

    q = vstart / arma::norm(vstart,2);
    Q = q;

    u_int i;
    for (i = 0; i < maxArSpace; ++i)
    {
        // build r , r = A * q
        applyUPDN(q, r);

        // orthogonalization step
        // orthogonalization is iterated for a more accurate result
        h = Q.t() * r;
        r = r - Q * h;
        c = Q.t() * r;
        r = r - Q * c;
        h = h + c;

        // creating the matrix T
        if (i == 0) T = h;
        else
        {
            T.resize(i+1,i+1);
            T(i,i-1) = hbefore;
            T.col(i) = h;
        }
        hbefore = arma::norm(r,2);

        // solve Aegean problem for T
        arma::eig_gen(eigenValsT, eigenVecsT, T);

        // eigenvals are not ordered
        // find the largest eigenvalue (according to abs)
        sorted_indeces = arma::sort_index(abs(eigenValsT),1);
        arma::vec errs = abs(eigenVecsT.row(i)).st() * hbefore;

        all_converged = true;
        for (u_int e = 0; e < min(i+1, number_of_eigs); ++e)
        {
            sorted_index = sorted_indeces(e);
            errors(e) = errs(sorted_index);
            if (errors(e) > 1.0e-14)
                all_converged = false;
        }


        if (hbefore < conv_thresh && -hbefore < conv_thresh) break;
        if (errors(number_of_eigs-1) > 1.0e-14) all_converged = false;
        if (all_converged) break;
        if (i+1 == maxArSpace) break;

        q = r/hbefore;
        Q = join_rows(Q,q);

    }

    for (u_int e = 0; e < number_of_eigs; ++e)
    {
        sorted_index = sorted_indeces(e);
        eigenValue(e) = eigenValsT(sorted_index);
        eigenVectors.col(e) = Q * eigenVecsT.col(sorted_index);
    }

    if (verbose)
    {
        lfout << "finished in : " << i << "steps" << std::endl;
        for (u_int i = 0; i < errors.size();++i)
            lfout << errors(i) << "\t";
        lfout << std::endl;
    }
    if (correlation_calculation)
    {
        lfout << "for correlation length calculation:" << std::endl;
        sorted_index = sorted_indeces(number_of_eigs);
        correlation_length.push_back(eigenValsT(sorted_index));
        lfout << eigenValsT(sorted_index) << " / " << eigenValue(0)
              << std::endl << std::endl;
    }
    return eigenValue(0);
}
