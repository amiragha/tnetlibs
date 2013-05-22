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
 * each Tensor contains a variable list of input and output
 * and also the cardinality of them
 * and a value
 */
typedef std::complex<double> cx_d;

class Tensor {
 public:
    std::vector<Index> indeces;
    std::vector<cx_d > values; /// complex<double> double matrix from armadillo package
    long allCards;
    int rank; /// rank of the Tensor (number of indeces)
    std::map<std::string, long> coeff;/// mapping from Index to the coefficient
    std::vector<int> vecCoeff; /// vector mapping from index to the coefficient

    // arma::cx_mat matRepresentation;

    Tensor(std::vector<Index> & indxs, std::vector<cx_d > & t);
    Tensor(std::vector<Index> & indxs);
    Tensor(){};
    ~Tensor();
    long prodCards();
    void print(int brk);
    arma::cx_mat toMat (const std::vector<Index> & rowIndeces,
                        const std::vector<Index> & colIndeces) const;
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

};


#endif /* _TENSOR_H_ */
