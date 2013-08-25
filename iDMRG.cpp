#include <iostream>
#include "iDMRG.h"
#include <iomanip>
#include <cmath>

using namespace std;
using namespace arma;

/**
 * constructors
 */
IDMRG::IDMRG(int mD){

    maxD = mD;

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

    // introducing the Heisenberg Hamiltonian matrices
    // TO-DO : find a better representation if possible
    cx_mat matHamilt = zeros<cx_mat>(10,10);
    matHamilt.submat(0,0,1,1) = I2;
    matHamilt.submat(8,8,9,9) = I2;
    matHamilt.submat(2,0,3,1) = PauliX;
    matHamilt.submat(8,2,9,3) = PauliX;
    matHamilt.submat(4,0,5,1) = PauliY;
    matHamilt.submat(8,4,9,5) = PauliY;
    matHamilt.submat(6,0,7,1) = PauliZ;
    matHamilt.submat(8,6,9,7) = PauliZ;

    cx_mat matHamilt_lmost = matHamilt.rows(8,9);
    cx_mat matHamilt_rmost = matHamilt.cols(0,1);

    // set the Hamiltonian and spin cardinalities
    B = 5;
    d = 2;

    /* the first dimension is always one
     * This is due to open boundary condition
     */
    matDims.push_back(1);

    // initialization of needed indeces for the whole class
    // TO-DO : find a better initialization for indeces
    bl.i("bl",B), br.i("br",B), b.i("b",B),
        sul.i("sul",d), sdl.i("sdl",d), sur.i("sur",d), sdr.i("sdr",d),
        sd.i("sd",d), su.i("su",d);

    // defining Hamiltonian Tensors
    // constructing Hamiltonian Tensors from matrices
    // TO-DO : check for potential wrong Index assignments
    Hamilt.fromMat(matHamilt, mkIdxSet(sd,bl), mkIdxSet(su,br));

    /* Note:
     * here WL has Indeces = sdl:d, sul:d, b:B
     * here WR has Indeces = sdr:d, b:B, sur:d
     * then W  has Indeces = sdl:d, sul:d, sdr:d, sur:d
     */
    WL.fromMat(matHamilt_lmost, mkIdxSet(sdl), mkIdxSet(sul,b));
    WR.fromMat(matHamilt_rmost, mkIdxSet(sdr,b), mkIdxSet(sur));
    W = WL * WR;
    W.rearrange(mkIdxSet(sdl,sdr,sul,sur));
    W.print(4);

    // TO-REMOVE
    //WL.print(2);
    //WR.print(10);

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
    double threshold = 1.0e-15;
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

    // TO-REMOVE : after checking
    cout << "eigenvec0: " << endl << eigenvecs.col(0) << endl;
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

    // lambda size check
    nextD = lambda_size_trunc(S);

    if (verbose)
        cout << "nextD = " << nextD << "   ";
    // for (nextD = 0; nextD < S.size(); ++nextD){
    //     if (less_thresh_test(S[nextD]))
    //         break;
    // }
    // Ds.push_back(nextD);

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
    cx_mat guessmat = randu<cx_mat>(2*nextD,2*nextD);
    guess.fromMat(guessmat, mkIdxSet(sul,lu), mkIdxSet(sur,ru));

    // update WL, WR and W to their final value
    /*
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
 * managing the first step which is solving for the four site lattice
 */
void
IDMRG::do_step(bool verbose){
    cout << endl;
    iteration++;
    if (verbose)
        cout << "Starting iteration number : " << iteration << endl;

    int nextD, D = matDims.back();

    // TO-REAMOVE
    //cout << "here W has the indeces : sdl bl sul sdr sur br" << endl;
    //W.printIndeces();

    cx_mat ksiVec = Lanczos();
    //cout << "ksiVec" << endl << ksiVec << endl;

    Index lu("lu", D), ru("ru", D),ld("ld", D), rd("rd", D);
    //cout << "left right indeces and w " << endl;
    // Left.reIndex(mkIdxSet(lu,bl,ld));
    // Right.reIndex(mkIdxSet(ru,br,rd));
    // //Left.printIndeces();
    // //Right.printIndeces();
    // //W.printIndeces();
    // Tensor HH = Left * W * Right;
    // // indeces of HH = lu,ld,sdl,sul,sdr,sur,ru,rd
    // cx_mat matHH = HH.toMat(mkIdxSet(sdl,ld,sdr,rd),
    //                         mkIdxSet(sul,lu,sur,ru));
    // cx_mat eigenvecs;
    // vec eigenvals;
    // eig_sym(eigenvals, eigenvecs, matHH);
    // // cout << "eigenvals= " << eigenvals <<endl;
    // cout << setprecision(15) <<"groundstate energy = " <<
    //     eigenvals(0)/(8*(iteration+1)) << endl;
    // cout << setprecision(15) << "fidelity of exact and lanczos :  " <<
    //     abs(cdot(eigenvecs.col(0),ksiVec)) << endl;

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

    // updating Left and Right
    update_LR(U_trunc, V_trunc, D, nextD);
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
    // qr(newB, left_lambda, lft.st());
    // newB = newB.t();
    // left_lambda = left_lambda.t();
    //svd
    svd(u, s, newB, lft);
    newB = newB.t();
    diags = diagmat(s);
    diags.resize(D,d*nextD);
    left_lambda = u*diags;
    // cout << u << endl;
    // cout << diags << endl;
    // cout << v << endl;


    /*
     * rotate right
     * note: S_trunc * V.t()_trunc is a matrix of nextD*D.d dimension
     * we need to perform svd on the matrix d.nextd*D
     */
    cout << "starting right rotation1" << endl;
    cx_mat rgt = (S * V.t()).st();
    rgt.reshape(D, d*nextD);
    // qr(newA, right_lambda, rgt.st());
    //svd
    svd(newA,s,v,rgt.st());
    //v = v.cols(0, D-1);
    diags = diagmat(s);
    diags.resize(nextD*d,D);
    right_lambda = diags*v.t();
    // cout << u << endl;
    // cout << diags << endl;
    // cout << v << endl;


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
    cout << "guess calculation" << endl;
    mat guesscore = inv(diagmat(lambda[lambda.size()-2]));

    // cout << "sizes : " << endl;
    // cout << newA.n_rows << ", "<< newA.n_cols <<endl;
    // cout << right_lambda.n_rows << ", "<< right_lambda.n_cols <<endl;
    // cout << guesscore.n_rows << ", "<< guesscore.n_cols <<endl;
    // cout << left_lambda.n_rows << ", "<< left_lambda.n_cols <<endl;
    // cout << newA.n_rows << ", "<< newB.n_cols <<endl;

    cx_mat guessMat = newA * right_lambda * guesscore * left_lambda * newB;
    //cout << "guessMat" << guessMat << endl;
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

    Left.reIndex(mkIdxSet(plu,b,pld));
    Right.reIndex(mkIdxSet(pru,b,prd));
    // constructing the Left and Right
    cout << "constructing Left and Right" << endl;
    Left = ((Left * A) * WL) * Astar;
    Left.printIndeces();
    Right = ((Right * B) * WR) * Bstar;
    Right.printIndeces();
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
    //cout << "guess" << endl;
    //guess.print(4);
    //cout << " r " << r << endl;

    q = r/norm(r,2);
    trial = q;
    r = operateH(q);
    //cout << " r " << r << endl;
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
        //cout << "T at level " << i << endl << T << endl;

        // calculating the eigenvalues of T
        eig_sym(eigenvals, eigenvecs, T);
        //cout << "eigenvals" << endl << eigenvals << endl;
        //cout << "eigenvecs" << endl << Q * eigenvecs << endl;

        // Error estimation
        // beta * eigenvecs last row
        // cout << "Error in eigenvalues" << endl
        //      << betas[i+1] * eigenvecs.row(i+1) << endl;
        // adding convergence test
        //error = betas[i+1] * eigenvecs(i+1,i+1);
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
 * iterate
 * iterate to the convergence
 */

void
IDMRG::iterate(){
    int N = 100;
    for (int iter=0; iter < N; ++iter){
        do_step(true);
    }

    cout << endl;

    // // printing fidelity
    // cout << "FIDELITY results" << endl;
    // for (int i = 0 ; i < fidelity.size(); ++i)
    // cout << endl;

    // printing energy results at each level
    cout << "ENERGY , fidelity, truncation, D results" << endl;
    for (int i = 0 ; i < energy.size(); ++i) {
        cout << setprecision(10) << energy[i]/(8.0*(i+1)) << "\t\t";
        cout << setprecision(10) << fidelity[i] << "\t\t";
        cout << setprecision(10) << lambda_truncated[i] << "\t\t";
        cout << setprecision(10) << matDims[i]<< "\t";
	cout << endl;
    }
    cout << endl;

    // printing Ds

    // cout << "ENERGY results" << endl;
    // for (int i = 0 ; i < energy.size(); ++i)
    // cout << endl;

}
