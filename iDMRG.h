/**
 * @file iDMRG.h
 */
#include "Tensor.h"
#include <fstream>

#ifndef _IDMRG_H_
#define _IDMRG_H_

/**
 * class iDMRG
 * this class contains an iDMRG precedure
 */

class IDMRG {

    bool verbose;
    std::ofstream lfout;
    int iteration; /// current iteration number
    u_int maxD; /// maximum size for matrices
    u_int B; /// Hamiltonian cardinalitis
    u_int d; /// spin cardinality
    std::vector<u_int> matDims; /// D at each level
    /*
     * Notes for Tensor indeces:
     * - Hamilt ,
     *     a general Tensor with Indeces = sd:d, bl:B, su:d, br:B
     * - W ,
     *     the hamiltonian for two site, always made by the contraction of
     *     WL * WR whith Indeces =
     */
    Tensor Hamilt; /// single cite H Tensor
    Tensor Left, Right; /// known Left and Right tensors
    Tensor guess; /// the guess Tensor
    Tensor WL, WR, W;
    Tensor UP_tensor; /// canonicalization
    Tensor DN_tensor; /// canonicalization
    std::vector<double> lambda_truncated; /// truncations
    std::vector<arma::vec> lambda; /// lambdas
    std::vector<double> fidelity;
    std::vector<double> energy;
    std::vector<double> convergence;
    bool converged;
    double convergence_threshold;
    double largestEV;
    Tensor canonical_Gamma;

public:
    arma::vec canonical_Lambda;
    /**
     * constructors
     */
    IDMRG(arma::cx_mat & mHamilt, u_int Bdim, u_int dim, u_int mD,
          double con_thresh = 1.0e-9, bool verbose = false,
          std::string logfile = "iDMRG_logfile.log");
    ~IDMRG();

private:

    /*
     * needed Indexes
     */
    Index bl,br,sul,sdl,sur,sdr,sd,su,b;

    /**
     * lambda_size_trunc
     * given the vector of lambdas, it will check them for small values and
     * find the next Dimension : nextD
     *
     * param S vector of not yet truncated lambdas
     *
     * return int nextD
     */
    u_int lambda_size_trunc (const arma::vec & S);

    /**
     * zeroth_iter
     * managing the zeroth step which are solving for the
     * two site lattice
     */
    void zeroth_iter();

    /**
     * do_step
     * go one step forward in the iDMRG algorithm
     */
    void do_step();

    /**
     * guess_calculate
     * given the truncated and ready to use U, V, S(mat), will rotate the center
     * and correctly calculated a guess (trial) tensor for lanczos to use
     */

    void guess_calculate(const arma::cx_mat & U, const arma::cx_mat & V,
                         const arma::mat & S, u_int D, u_int nextD);
    /**
     * update_LR
     * given the new A and B, updates Left and Right matrices
     */
    void update_LR(const arma::cx_mat & U, const arma::cx_mat & V,
                   u_int D, u_int nextD);

    /**
     * canonicalize
     * canonicalize the wavefunction using the middle A,B, lambda calculated
     */
    void canonicalize( Tensor A, Tensor B, u_int D, u_int nextD);

    /**
     * Lanczos
     * given a guess vector and a reference ksi, updates the ksi to the
     * eigenvector with smallest eigenvalue of L*W*R
     * input guess Tensor and the ksi vector as reference
     *
     * return void
     */
    arma::cx_vec Lanczos();


    /**
     * operateH
     * find the effect of bigH(Hamiltonians and Left and Right) on a given
     * vector
     */
    arma::cx_vec operateH(arma::cx_vec & q);


    /**
     * arnoldi_canonical
     * performs arnoldi algorithms using UP_tensor and DN_tensor
     * a part of canonicalization process
     *
     */
    cx_d arnoldi_canonical(Tensor & V);

public:
    /**
     * iterate
     * Iterate to the convergence
     */
    void iterate();

    /**
     * Renyi entroypy calculator
     */
    double renyi(double alpha, const arma::vec & L);
};

#endif /* _IDMRG_H_ */
