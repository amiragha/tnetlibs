#include "ternaryMera.h"

void giveRandomDensity (Tensor & DensityMatrix, int card){

    Index d1("d1",card),d2("d2",card),d3("d3",card),d4("d4",card);
    std::vector<Index> d_idcs = {d1 ,d2, d3, d4};
    std::vector<Index> dr1  = {d1};
    std::vector<Index> dc1  = {d2};
    std::vector<Index> dr2  = {d3};
    std::vector<Index> dc2  = {d4};
    std::vector<Index> dvec1 = {d1,d2};
    std::vector<Index> dvec2 = {d3,d4};

    arma::cx_mat X = arma::randu<arma::cx_mat>(card,1);
    arma::cx_mat U, V;
    arma::vec s;
    svd(U,s,V,X);
    Tensor U1(dvec1), U2(dvec2);
    //std::cout <<"U" <<std::endl << U << std::endl;
    //std::cout <<"V" <<std::endl << V << std::endl;
    U1.fromMat(U,dr1,dc1);
    U2.fromMat(U,dr2,dc2);
    U2.conjugate();
    //std::cout << U1.indeces.size() << std::endl;
    //std::cout << U2.indeces.size() << std::endl;
    //U1.print(card*card);
    //U2.print(card*card);

    DensityMatrix = (U1*U2)/card;
}
