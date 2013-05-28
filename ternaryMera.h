/**
 * @file ternaryMera.h
 */
#include "Tensor.h"
#include <cstdlib>
#include <ctime>

#ifndef _TERNARYMERA_H_
#define _TERNARYMERA_H_

/**
 * class ternaryMera
 * this class contains all of necessary Tensor and functions for
 * operating ternary Mera
 */

class TernaryMera {
    int numlevels; /// maximum cardinality for Tensor indeces
    double thresh;
    std::vector<int> cards; /// holding cardinalities for each level
    std::vector<Tensor> Hamiltonian; /// holding Hamiltonians
    std::vector<Tensor> Isometry; /// holding Isometries
    std::vector<Tensor> Unitary; /// holding Unitaries
    std::vector<Tensor> DensityMatrix; /// holding DensityMatrices

 public:

    /**
     * constructors
     */
    TernaryMera(arma::cx_mat & Hmat, int icard, int fcard,
                bool verbose = false);
    ~TernaryMera();

 private:

    /**
     * give_random_UniIso
     * find a random Unitary and Isometry, called during initialization
     *
     * return void
     */
    void give_random_UniIso(int level);

    /** give_random_density
     * find a random DensityMatrix satisfying all the properties
     * called during initialization
     *
     * return void
     */
    void give_random_density();

 public:

    /**
     * ascend
     * performing ascending for the  Hamiltonian at level with corresponding
     * Isometries and Unitaries, changes the Hamiltonian at level+1
     *
     * param level an int showing the wanted level
     *
     * return void
     */
    void ascend (int level, bool verbose = false);

    /**
     * descend
     * performing descending of the DensityMatrix at level with corresponding
     * Isometris and Unitaries, changes the DensityMatrix at level-1
     *
     * param level an int showing the wanted level
     *
     * return void
     */
    void descend (int level, bool verbose = false);

    /**
     * iso_env
     * finds the environment of an Isometry at level using Hamiltonian and
     * Unitary at the same level and DensityMatrix at level+1
     *
     * param level an int
     *
     * return Tensor of environment for isometry
     */
    Tensor iso_env (int level, bool verbose = false, bool negateH = true);

    /**
     * uni_env
     * finds the environment of a Unitary  at level using Hamiltonian and
     * Isometry at the same level and DensityMatrix at level+1
     *
     * param level an int
     *
     * return Tensor of environment for unitary
     */
    Tensor uni_env (int level, bool verbose = false, bool negateH = true);

    /**
     * arnoldi
     * performing arnoldi algorithm for the descend operator and finding
     * the fixed point DensityMatrix, changes DensityMatrix at the last level
     *
     * return void
     */
    void arnoldi(bool verbose = false);

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
    double energy (int level);
    std::vector<double> energy (bool verbose = false);

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
    void iso_update (int level, int num_update,
                     bool verbose = false, bool negateH = true);

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
    void uni_update (int level, int num_update,
                     bool verbose = false, bool negateH = true);

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
    void bottom_up(bool verbose = false, bool negateH = true);
};
#endif /* _TERNARYMERA_H_ */
