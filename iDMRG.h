/**
 * @file iDMRG.h
 */
#include "Tensor.h"

#ifndef _IDMRG_H_
#define _IDMRG_H_

/**
 * class iDMRG
 * this class contains an iDMRG precedure
 */

class IDMRG {

    int iteration; /// current iteration number
    int maxD; /// maximum size for matrices
    int B; /// Hamiltonian cardinalitis
    int d; /// spin cardinality
    std::vector<int> matDims; /// D at each level
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
    std::vector<double> lambda_truncated; /// truncations
    std::vector<arma::vec> lambda; /// lambdas
    std::vector<double> fidelity;
    std::vector<double> energy;

public:
    /**
     * constructors
     */
    IDMRG(int mD);
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
    int lambda_size_trunc (const arma::vec & S);

    /**
     * zeroth_iter
     * managing the zeroth step which are solving for the
     * two site lattice
     */
    void zeroth_iter(bool verbose = false);

    /**
     * update_LR
     * given A and B this function will update Left and Right
     */
    /**p
     * first_iter
     * managing the first step which is solving for the four site lattice
     */
    void do_step(bool verbose = false);

    /**
     * guess_calculate
     * given the truncated and ready to use U, V, S(mat), will rotate the center
     * and correctly calculated a guess (trial) tensor for lanczos to use
     */
    void guess_calculate(const arma::cx_mat & U, const arma::cx_mat & V,
                         const arma::mat & S, int D, int nextD);
    /**
     * update_LR
     * given the new A and B, updates Left and Right matrices
     */
    void update_LR(const arma::cx_mat & U, const arma::cx_mat & V,
		   int D, int nextD);

public:

    /**
     * operateH
     * find the effect of bigH(Hamiltonians and Left and Right) on a given
     * vector
     */
    arma::cx_vec operateH(arma::cx_vec & q);

    /**
     * Iterate
     * Iterate to the convergence
     */
    void go_one_step(bool verbose = false);
    void iterate();

    /**
     * Lanczos
     * given a guess vector and a reference ksi, updates the ksi to the
     * eigenvector with smallest eigenvalue of L*W*R
     * input guess Tensor and the ksi vector as reference
     *
     * return void
     */
    arma::cx_vec Lanczos();
};

#endif /* _IDMRG_H_ */