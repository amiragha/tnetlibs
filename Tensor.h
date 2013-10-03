/**
 * @file Tensor.h
 */
#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <vector>
#include <string>
#include <map>
#include <armadillo>
#include "Index.h"
#include <complex>

/**
 * Tensor class
 * indeces:= a vector of all the indeces
 * coeff:= mapping from index to coefficients (map)
 * vecCoeff:= vector of coefficient for indeces
 * allCards:= is the product of all the cardinalities
 * values:= values stored in the Tensor
 * rank:= rank of the the Tensor
 */
typedef std::complex<double> cx_d;

class Tensor {
public:
    std::vector<Index> indeces; /// vector of Indexes of the Tensor
    std::vector<cx_d> values; /// complex<double> vector of values
    long allCards; /// product of all cardinalities
    int rank; /// rank of the Tensor (number of indeces)
    std::map<std::string, long> coeff;/// mapping from Index to the coefficient
    std::vector<int> vecCoeff; /// vector mapping from index to the coefficient

    /**
     * constructors
     */
    Tensor(std::vector<Index> & indxs, std::vector<cx_d > & t);
    Tensor(std::vector<Index> & indxs);
    Tensor(){};
    ~Tensor();

    /**
     * prodCards
     * finding the allCards of the tensor and filling the cardinality table
     * in the coeff.
     *
     * return the full cardinality of the Tensor
     */
    long prodCards();

    /**
     * printing the tensor
     * receiving an int for number of elements to print on one line
     * param brk number of elements before a line break
     * does a pretty printing of complex numbers
     * return void
     */
    void print(int brk);

    /**
     * toMat
     * creating a cx_mat from a tensor give the indeces to put on the row
     * and column of the resulting matrix
     *
     * param rowIndeces indeces to keep in row
     * param colIndeces indeces to keep in column
     *
     * return cx_mat the resulting matrix
     */
    arma::cx_mat toMat (u_int num_row, u_int num_col) const;

    arma::cx_mat toMat (const std::vector<Index> & rowIndeces,
                        const std::vector<Index> & colIndeces) const;

    arma::cx_mat& toMat_aux (const std::vector<Index> & rowIndeces,
                             const std::vector<Index> & colIndeces,
                             arma::cx_mat& result) const;
    /**
     * fromMat
     * creating a tensor from a cx_mat matrix given the indeces that are on the row
     * and column of the given matrix.
     *
     * param matrix the cx_mat
     * param rowIndeces indeces to keep in row
     * param colIndeces indeces to keep in column
     *
     * return void
     */
    void fromMat(const arma::cx_mat & matrix,
                 const std::vector<Index> &row, const std::vector<Index> & col);

    /**
     * toVec
     * creating a cx_vec from a Tensor
     *
     * return cx_vec the resulting vector
     */
    arma::cx_vec toVec ();

    /**
     * fromVec
     * creating a tensor from cx_vec vector given the inedeces
     *
     * param vector cx_vec
     * param vecIndeces
     *
     * return void
     */
    void fromVec(const arma::cx_vec & vector,
                 const std::vector<Index> & vecIndeces);
    /**
     * similarities
     *
     * finding the similarities between indeces of this tensor with another one
     *
     * param other Tensor
     *
     * return a vector consisting of similar indeces (contracting), other
     * indeces of this Tensor as row of the final matrix and other indeces
     * of other Tensor as col of the final matrix.
     */
    std::vector<std::vector<Index> > similarities(const Tensor &other);

    /**
     * overloading operator *
     *
     * the * operator does the tensor product of the two tensors
     * if there is any common indeces it manages the contraction
     *
     * it first finds the final indeces by calling the similarities function
     * and then performs the product-contraction operation
     *
     * return a new Tensor
     */
    Tensor operator * (const Tensor & other);
    Tensor operator + (const Tensor & other);
    Tensor& operator / (double num);

    /**
     * conjugate
     * taking the complex conjugate of all the element of the Tensor
     *
     * changes the current Tensor
     * return Tensor conjugated of the same Tensor
     */
    Tensor conjugate();

    /**
     * reIndex
     * changing the Indeces of the Tensor while leaving the elements unchanged
     * this correspond to just renaming the Indeces.
     * note: overloaded to receive 4 input Indeces for ease of use with rank 4
     * indeces which happens to occur in our problem
     * changes the Tensor indeces
     *
     * param vector<Index> & new newIndeces or 4 Indexes
     *
     * return void
     */
    void reIndex(const Index a1, const Index a2,
                 const Index a3, const Index a4);
    void reIndex(const std::vector<Index> & newIndeces);
    void rearrange(const std::vector<Index> & newOrder);
    void printIndeces() const;

    /**
     * mapFinder
     * finding the mapping between indeces with indexes in the full vector
     * param fullIndeces a vector of indexes contating all of the indeces
     *
     * return vector<int> that is the indexes of indeces in the full vector
     */
    std::vector<int> mapFinder(const std::vector<Index> fullIndeces) const;

    /**
     * getValueOfAsgn
     * getting the value to the given assignment for indeces
     *
     * param asgns is a the given assignments (vector<int>)
     *
     * return complex<double> or cx_d
     */
    cx_d getValueOfAsgn(const std::vector<int> asgns) const;

    /**
     * slice
     * slicing a Tensor in one index
     *
     * param index index to be sliced
     * param from start
     * param upto end
     *
     * return Tensor a sliced new Tensor
     */
    Tensor slice(Index index, u_int from, u_int upto);
};


#endif /* _TENSOR_H_ */
