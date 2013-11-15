#include "IDMRGS.h"
#include <algorithm>
#include <cassert>

void IDMRGS::Init()
{
    // sort mFidDeltas
    std::sort (mFidDeltas.begin(), mFidDeltas.end());
    std::reverse (mFidDeltas.begin(), mFidDeltas.end());
    for (u_int idx = 0; idx < mFidDeltas.size(); ++idx)
        std::cout << mFidDeltas[idx] << std::endl;
    std::cout << std::endl;

    // calculate mJadds
    mJadds = mFidDeltas;
    mJadds.push_back(0.0);
    for (u_int idx = 0; idx < mFidDeltas.size(); ++idx)
    {
        mJadds[idx] /= -2.0;
        mJadds.push_back(-mJadds[idx]);
    }
    for (u_int idx = 0; idx < mJadds.size(); ++idx)
        std::cout << mJadds[idx] << std::endl;
    std::cout << std::endl;

    // open the result file and print useful
    mDataOut.open(mDataFile.c_str());
    mDataOut << "import numpy as np"    << std::endl
             << "Ent_Spectrum = dict()" << std::endl
             << "Energy = dict()"       << std::endl;

    for (u_int idx = 0; idx < mOneSites.size(); ++idx)
        mDataOut << mOneSites[idx].first << " = dict()" << std::endl;

    if (mFidDeltas.size() > 0)
    {
        mDataOut << "Energy2nd = dict()"    << std::endl
                 << "Fid = dict()"          << std::endl;
    }

}

void IDMRGS::Go()
{
    std::cout << "inside Go" << std::endl;
    arma::cx_mat mat_hamilt;
    u_int n_fid = mJadds.size();
    u_int n_middle = (n_fid - 1)/2;
    IDMRG** idmrgs = new IDMRG*[n_fid];

    // first IDMGS instance
    double current_j = mJs[0] + mJadds[0];
    mat_hamilt = mHamiltonian(current_j);
    idmrgs[0] = new IDMRG(mat_hamilt, mBdim, mSdim, mMaxD, CONV);

    Tensor Left        = idmrgs[0]->get_Left();
    Tensor Right       = idmrgs[0]->get_Right();
    arma::cx_vec guess = idmrgs[0]->get_guess();
    arma::vec llamb    = idmrgs[0]->get_llamb();

    // other instances
    for (u_int j_idx = 0; j_idx < mJs.size(); ++j_idx)
    {
        // create array of pointers to Idmrg for different mFidDeltas
        for (u_int f_idx = 0; f_idx < n_fid; ++f_idx)
        {
            if (j_idx == 0 && f_idx == 0) continue;

            // find the j
            current_j = mJs[j_idx] + mJadds[f_idx];
            mat_hamilt = mHamiltonian(current_j);

            // Idmrg
            idmrgs[f_idx] = new IDMRG(mat_hamilt, mBdim, mSdim, mMaxD,
                                      Left, Right, guess, llamb, CONV);
            // change left, right, guess, llamb
            Left  = idmrgs[f_idx]->get_Left();
            Right = idmrgs[f_idx]->get_Right();
            guess = idmrgs[f_idx]->get_guess();
            llamb = idmrgs[f_idx]->get_llamb();

        }
        current_j = mJs[j_idx];

        // simple reports for n_middle
        // reporting onesites
        for (u_int idx = 0; idx < mOneSites.size(); ++idx)
        {
            mDataOut << mOneSites[idx].first << "[("<< mMaxD << ","
                     << current_j << ")]= "
                     << idmrgs[n_middle]->
                expectation_onesite(mOneSites[idx].second)
                     << std::endl;
        }
        // reporting energy
        double middle_energy = idmrgs[n_middle]->FinalEnergy();
        mDataOut << "Energy[(" << mMaxD << "," << current_j << ")]= "
                 << middle_energy << std::endl;

        // reporting entanglement spectrum
        mDataOut << "Ent_Spectrum[(" << mMaxD << "," << current_j
                 << ")] = np.array([" << std::endl;
        u_int es_size = idmrgs[n_middle]->entanglement_spectrum.size();
        for (u_int idx = 0; idx < es_size; ++idx)
        {
            mDataOut << "    " <<
                idmrgs[n_middle]->entanglement_spectrum(idx)
                     << "," << std::endl;
        }
        mDataOut << "])" << std::endl;
        // compute and report the GsFidelity and second derivative of Energy
        if (n_fid > 1)
        {
            for (u_int l=0; l < n_middle; ++l)
            {
                double delta = mFidDeltas[l];
                std::cout << "delta = " << delta << std::endl;
                u_int r = n_fid - l - 1;

                // energy second derivative
                mDataOut << "Energy2nd[(" << mMaxD << "," << current_j
                         << "," << delta  << ")] = "
                         << (idmrgs[l]->FinalEnergy() +
                             idmrgs[r]->FinalEnergy()
                             - 2 * middle_energy)/(delta * delta)
                         << std::endl;

                // calculate gs fidelitly
                mUp = idmrgs[l]->get_GL();
                mDn = idmrgs[r]->get_GL();
                arma::vec gs_fid = GsFidelity();

                // report fidelity
                mDataOut << "Fid[(" << mMaxD << "," << current_j << ","
                         << delta << ")] = np.array([" << std::endl;
                for (u_int f = 0; f < gs_fid.size(); ++f)
                    mDataOut << "    " << gs_fid(f) << "," << std::endl;
                mDataOut << "])" << std::endl << std::endl;
            }
        }

        // deletes
        for (u_int f_idx = 0; f_idx < n_fid; ++f_idx)
            delete idmrgs[f_idx];

        std::cout << "DONE for maxD = " <<
            mMaxD << " , J = " << current_j << std::endl;
    }

    delete [] idmrgs;
    // finalize the mDatafile
    mDataOut << std::endl
             << "if __name__ == \"__main__\" :" << std::endl
             << "    main()" << std::endl;
    mDataOut.close();
}

arma::vec IDMRGS::GsFidelity(){
    // defining D
    u_int Dl = mUp.indeces[0].card;
    u_int Dr = mDn.indeces[0].card;
    u_int D = (Dl > Dr) ? Dr : Dl;
    std::cout << "Dl = " << Dl << ", Dr = " << Dr
              << " => D =" << D << std::endl;

    Index vu("vu", D), vd("vd", D),
        lu("lu", D), ru("ru", D), ld("ld", D), rd("rd",D);

    mDn = mDn.conjugate();
    Index sul = mUp.indeces[1];
    Index sur = mUp.indeces[2];
    // asserting the equality of sur, sul in mDn/mUp tensors
    assert (sul == mDn.indeces[1]);
    assert (sur == mDn.indeces[2]);

    // needed truncation
    mUp = mUp.slice(mUp.indeces[0],0,D).slice(mUp.indeces[3],0,D);
    mDn = mDn.slice(mDn.indeces[0],0,D).slice(mDn.indeces[3],0,D);

    mUp.reIndex(lu, sul, sur, vu);
    mDn.reIndex(ld, sul, sur, vd);

    arma::cx_vec vstart = arma::randu<arma::cx_vec>(D*D);
    return abs(Arnoldi(vstart,1));
}


arma::cx_vec IDMRGS::Arnoldi(const arma::cx_vec& vstart, u_int num)
{
    double conv_thresh = 1.e-15;
    arma::cx_vec eigenValue(num);
    u_int maxArSpace = vstart.size();
    u_int number_of_eigs = num;

    // defining needed variables
    arma::cx_vec         q, r, h, c,eigenValsT;
    arma::cx_mat         Q, eigenVecsT;
    arma::cx_mat         T;
    arma::vec            errors;
    arma::uvec           sorted_indeces;
    arma::uword          sorted_index;
    double               hbefore = 0.0;
    bool                 all_converged = false;
    //u_int                vectorDim = vstart.size();

    // initialize
    eigenValue   = arma::zeros<arma::cx_vec>(number_of_eigs);
    errors       = arma::ones<arma::vec>(number_of_eigs);

    q = vstart / arma::norm(vstart,2);
    Q = q;

    u_int i;
    for (i = 0; i < maxArSpace; ++i)
    {
        // build r , r = A * q
        ApplyUpDn(q, r);

        // orthogonalization step
        // orthogonalization is iterated for a more accurate result
        h = Q.t() * r;
        r = r - Q * h;
        c = Q.t() * r;
        r = r - Q * c;
        h = h + c;

        // creating the matrix T
        if (i == 0) T = h;
        else
        {
            T.resize(i+1,i+1);
            T(i,i-1) = hbefore;
            T.col(i) = h;
        }
        hbefore = arma::norm(r,2);

        // solve Aegean problem for T
        arma::eig_gen(eigenValsT, eigenVecsT, T);

        // eigenvals are not ordered
        // find the largest eigenvalue (according to abs)
        sorted_indeces = arma::sort_index(abs(eigenValsT),1);
        arma::vec errs = abs(eigenVecsT.row(i)).st() * hbefore;

        all_converged = true;
        for (u_int e = 0; e < min(i+1, number_of_eigs); ++e)
        {
            sorted_index = sorted_indeces(e);
            errors(e) = errs(sorted_index);
            if (errors(e) > 1.0e-14)
                all_converged = false;
        }


        if (hbefore < conv_thresh && -hbefore < conv_thresh) break;
        if (errors(number_of_eigs-1) > 1.0e-14) all_converged = false;
        if (all_converged) break;
        if (i+1 == maxArSpace) break;

        q = r/hbefore;
        Q = join_rows(Q,q);

    }

    for (u_int e = 0; e < number_of_eigs; ++e)
    {
        sorted_index = sorted_indeces(e);
        eigenValue(e) = eigenValsT(sorted_index);
    }
    return eigenValue;
}

void IDMRGS::ApplyUpDn(const arma::cx_mat& input, arma::cx_mat& output)
{
    u_int D = mUp.indeces[0].card;
    assert (D*D == input.size());
    Index vu("vu",D),vd("vd",D);
    Tensor V;
    V.fromVec(input, mkIdxSet(vu,vd));
    V = (V * mUp) * mDn;
    output = V.toVec();
}
