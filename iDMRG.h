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
typedef arma::cx_mat (*MpoType) (double J);
class IDMRG
{
public:
    /**
     * constructors
     */
    IDMRG(arma::cx_mat& mHamilt, u_int Bdim, u_int dim, u_int mD,
          double con_thresh = 1.0e-9, bool verbose = false,
          std::string logfile = "iDMRG_logfile.log");
    IDMRG(arma::cx_mat& mHamilt, u_int Bdim, u_int dim, u_int mD,
          Tensor& in_Left, Tensor& in_right,
          arma::cx_vec& in_guess, arma::vec& in_llamb,
          double con_thresh = 1.0e-9, bool verbose = false,
          std::string logfile = "iDMRG_logfile.log");
    ~IDMRG();
    cx_d arnoldi(arma::cx_vec& vstart,
                 arma::cx_mat& eigenVectors,
                 bool correlation_calculation = false);
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

    /**
     * effect of symmetry
     */
    double SymmetryEffect(arma::cx_mat symmetry_op);

    // ACCESSES
    Tensor get_GL() const;
    Tensor get_LG() const;
    Tensor get_LGL() const;
    Tensor get_Gamma() const;
    Tensor get_Left() const;
    Tensor get_Right() const;
    arma::cx_vec get_guess() const;
    arma::vec get_llamb() const;

    arma::vec get_Lambda() const;

    std::vector<cx_d> correlation_length;
    double FinalEnergy() {return mFinalEnergy; }
private:
    bool                         verbose;
    u_int                        iteration;
    std::ofstream                lfout;
    u_int                        B, d, maxD, finalD;
    std::vector<u_int>           matDims; /// D at each level
    arma::cx_mat                 energyMPO;

    // needed Tensor and matrices for iDMRG calcualation
    arma::cx_vec                 guess;
    Tensor                       Hamilt, WL, WR, W;
    Tensor                       Left, Right;
    Tensor                       UP_tensor, DN_tensor; /// canonicalization
    std::vector<arma::vec>       lambda;
    arma::vec                    llamb;

    // iteration information for reporting
    std::vector<double>          truncations;
    std::vector<double>          guessFidelity;
    std::vector<double>          energy;
    std::vector<double>          convergence;
    double                       largestEV;

    // convergence related
    bool                         converged;
    double                       convergence_threshold;

    // final results of iDMRG calcualtion
    Tensor                       canonical_Gamma;
    arma::vec                    canonical_Lambda;
    double                       mFinalEnergy;

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
    void zeroth_iter_with_init();
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
    double Lanczos(arma::cx_vec& guess, arma::cx_vec& result);


    /**
     * operateH
     * find the effect of bigH(Hamiltonians and Left and Right) on a given
     * vector
     */
    void operateH(arma::cx_vec & q, arma::cx_vec & res);


    /**
     * applyUPDN
     */
    void applyUPDN(const arma::cx_vec & in , arma::cx_vec & out);
};

arma::vec gsFidelity(const IDMRG & left, const IDMRG & right);
#endif /* _IDMRG_H_ */
