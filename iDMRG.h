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
    arma::vec canonical_Lambda;

public:
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
     * applyUPDN
     */
    void applyUPDN(const arma::cx_vec & in , arma::cx_vec & out);
public:
    cx_d arnoldi(arma::cx_vec& vstart,
                 arma::cx_mat& eigenVectors);
    /**
     * arnoldi_canonical
     * performs arnoldi algorithms using UP_tensor and DN_tensor
     * a part of canonicalization process
     *
     */
    cx_d arnoldi_canonical(Tensor & V);

    arma::vec entanglement_spectrum;
    /**
     * iterate
     * Iterate to the convergence
     */
    void iterate();

    /**
     * get_canonical_lambda
     * gives canonical lambda which sums to one
     */

    /**
     * expectation_onesite
     * calculates the expectation value of a given one-site operator
     * using canonical Lambda and Gammma
     * an example is S_z
     */
    double expectation_onesite(arma::cx_mat onesite_op);

    /**
     * expectation_twosite
     * calculates the expectation value of a given  two-site operator
     * using canonical Lambda and Gammma
     * an example is the energy for NN models
     */
    double expectation_twosite(arma::cx_mat twosite_op);

    /**
     * gsFidelity
     * calculates the ground state fidelity
     * given the MPO for left and right Hamiltonians, which are different
     * for a small amount of change in the desired parameter
     */
    double gsFidelity(arma::cx_mat leftmatHamilt,
                      arma::cx_mat rightmatHamilt);
    /**
     * Renyi entroypy calculator
     */
    double renyi(double alpha, const arma::vec & L);
    Tensor get_Gamma() const;
    arma::vec get_Lambda() const;
};

arma::cx_vec gsFidelity(const IDMRG & left, const IDMRG & right);
#endif /* _IDMRG_H_ */
