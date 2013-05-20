#include "ternaryMera.h"

void giveRandomTensors(Tensor & Unitary, Tensor & Isometry, int in, int out){

    //srand(time(NULL));
    Index u1("u1",in),u2("u2",in),u3("u3",in),u4("u4",in),
        t1("t1",in),t2("t2",in),t3("t3",in),t4("t4",out);

    // defining the Unitary and Isometry
    std::vector<Index> u_idcs;
    std::vector<Index> u_ridx;
    std::vector<Index> u_cidx;
    std::vector<Index> t_idcs;
    std::vector<Index> t_ridx;
    std::vector<Index> t_cidx;

    // using singular value decomposition
    arma::cx_mat X = arma::randu<arma::cx_mat>(in*in*in,in*in);
    arma::cx_mat U, V;
    arma::vec s;
    svd(U,s,V,X);
    // std::cout << "U:" << std::endl;
    // std::cout << U << std::endl;
    // std::cout << "V:" << std::endl;
    // std::cout << V << std::endl;

    u_idcs = {u1 ,u2, u3, u4};
    u_ridx = {u1, u2};
    u_cidx = {u3, u4};
    t_idcs = {t1 ,t2, t3, t4};
    t_ridx = {t1, t2, t3};
    t_cidx = {t4};

    Unitary.indeces = u_idcs;
    Unitary.fromMat(V,u_ridx,u_cidx);
    Isometry.indeces = t_idcs;
    Isometry.fromMat(U,t_ridx,t_cidx);

}
