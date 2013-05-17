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
    std::map<std::string, long> coeff;/// mapping from Index to the coefficient

    // arma::cx_mat matRepresentation;

    Tensor(std::vector<Index> & indxs, std::vector<cx_d > & t);
    Tensor(std::vector<Index> & indxs);
    Tensor(){};
    ~Tensor();
    long prodCards();
    void print(int brk);
    arma::cx_mat toMat (const std::vector<Index> & rowIndeces,
               const std::vector<Index> & colIndeces) const;
    void fromMat(const arma::cx_mat & matrix,
                 const std::vector<Index> &row, const std::vector<Index> & col);
    std::vector<std::vector<Index> > similarities(const Tensor &other);
    Tensor operator * (const Tensor & other);
    Tensor operator + (const Tensor & other);
    Tensor& operator / (double num);
    void operator << (cx_d num);
    void conjugate();
    void reIndex(const std::vector<Index> & newIndeces);
    void rearrange(const std::vector<Index> & newOrder);
    void printIndeces() const;
};


#endif /* _TENSOR_H_ */
