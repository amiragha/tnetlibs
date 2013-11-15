/** IDMRGS class
 *
 * this class contains iDMRG procedures on a interval with the
 * possibility of calculating the GS fidelity and the second
 * derivative of energy.
 */
#include "iDMRG.h"

#ifndef _IDMRGS_H_
#define _IDMRGS_H_

typedef arma::cx_mat (*MpoType) (double);
typedef std::vector<std::pair<std::string, arma::cx_mat> > OneSitesType;
class IDMRGS
{
public:
    // LIFECYCLE
    /** constructor
     */
    IDMRGS (MpoType                  mHamiltonian,
            std::vector<double>&     mJs,
            u_int                    mBdim,
            u_int                    mSdim,
            u_int                    mMaxD,
            OneSitesType             mOneSites,
            std::vector<double>      mFidDeltas = std::vector<double>(),
            double                   CONV = 1.0e-9,
            bool                     mVerbose = false,
            std::string              mDataFile = "result_IDMRGS.py",
            std::string              mLogFile  = "IDMRG_logfile.log")
    : mHamiltonian   (mHamiltonian),
        mJs            (mJs),
        mBdim          (mBdim),
        mSdim          (mSdim),
        mMaxD          (mMaxD),
        mOneSites      (mOneSites),
        mFidDeltas     (mFidDeltas),
        CONV           (CONV),
        mVerbose       (mVerbose),
        mDataFile      (mDataFile),
        mLogFile       (mLogFile)
    {Init(); Go();}

    /** destructor
     */
    ~IDMRGS() {}

private:
    MpoType                     mHamiltonian;
    std::vector<double>         mJs;
    u_int                       mBdim;
    u_int                       mSdim;
    u_int                       mMaxD;
    OneSitesType                mOneSites;
    std::vector<double>         mFidDeltas;
    std::vector<double>         mJadds;
    double                      CONV;
    bool                        mVerbose;
    std::string                 mDataFile;
    std::string                 mLogFile;
    std::ofstream               mDataOut;
    Tensor                      mUp, mDn;

    // OPERATIONS
    /** initialize iDMRG
     */
    void Init();

    /** Go calculating the iDRMG on the interval
     */
    void Go();

    /** GsFidelity
     */
    arma::vec GsFidelity();

    /** Arnoldi
     */
    arma::cx_vec Arnoldi(const arma::cx_vec& vstart, u_int num);

    /** ApplyUpDn
     */
    void ApplyUpDn(const arma::cx_mat& input, arma::cx_mat& output);

    /** miscellaneous
     */
    inline u_int min(u_int one, u_int two)
    {
        return (one > two) ? two : one;
    }
};

#endif /* _IDMRGS_H_ */
