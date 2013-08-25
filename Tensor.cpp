#include <iostream>
#include "Tensor.h"

using namespace std;
using namespace arma;

/**
 * Constructor Methods
 */
Tensor::Tensor(vector<Index> & indxs,
               vector<cx_d > & vals){

    indeces = indxs;
    values = vals;
    allCards = prodCards();
}

Tensor::Tensor(vector<Index> & indxs){
    indeces = indxs;
    allCards = prodCards();
    //values = vector<cx_d> (allCards, 0.0);
}

Tensor::~Tensor(){
    // cout << "dying" << endl;
}

/**
 * printing the tensor
 * receiving an int for number of elements to print on one line
 * param brk number of elements before a line break
 * does a pretty printing of complex numbers
 * return void
 */
void
Tensor::print(int brk){
    double real, imag;
    double thresh = 0.0000000001;
    for (int i = 0; i < values.size(); ++i)
    {
        if (i%brk ==0)
            cout<<endl;
        real = values[i].real();
        imag = values[i].imag();
        //cout << i << ": ";
        if (real < thresh && real > -thresh) real = 0.0;
        if (imag < thresh && imag > -thresh) cout <<real<<" ";
        else {
            if (imag < 0.0)
                cout <<real<<"-i"<<-imag<< " ";
            else
                cout <<real<<"+i"<<imag<<" ";
        }
    }
    cout << endl;
}

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
cx_mat
Tensor::toMat(int num_row, int num_col) const {
    if (num_row + num_col != indeces.size())
        cout << "ERROR: toMat, not equal to number of indeces!" << endl;
    vector<Index> rowIndeces (indeces.begin(), indeces.begin()+num_row);
    vector<Index> colIndeces (indeces.begin()+num_row, indeces.end());
    cx_mat resultMat;
    toMat_aux(rowIndeces,colIndeces, resultMat);
    return resultMat;
}

cx_mat
Tensor::toMat (const vector<Index> & rowIndeces,
               const vector<Index> & colIndeces) const {

    cx_mat resultMat;
    toMat_aux(rowIndeces,colIndeces, resultMat);
    return resultMat;
}

cx_mat&
Tensor::toMat_aux (const vector<Index> & rowIndeces,
                   const vector<Index> & colIndeces, cx_mat& result ) const {

    // defining and initilalizing matCoeff, matcards, matasgns
    vector<int> matcards, matasgns, matcoeff;
    int prod = 1, prodcol, prodrow;
    for (int i = 0; i < rowIndeces.size(); ++i)
    {
        matcards.push_back(rowIndeces[i].card);
        matasgns.push_back(0);
        matcoeff.push_back(coeff.at(rowIndeces[i].name));
        prod *= rowIndeces[i].card;
    }
    prodrow = prod;
    for (int i = 0; i < colIndeces.size(); ++i)
    {
        matcards.push_back(colIndeces[i].card);
        matasgns.push_back(0);
        matcoeff.push_back(coeff.at(colIndeces[i].name));
        prod *= colIndeces[i].card;
    }
    prodcol = prod/prodrow;

    result = cx_mat(prodrow, prodcol);

    int ridx, cidx;
    // initializing iterator
    vector<cx_d>::const_iterator iter = values.begin();

    for (int r = 0; r < prodrow; ++r)
    {
        for (int c = 0; c < prodcol; ++c)
        {
            // putting the value into the matrix
            result(r,c) = *iter;
            // updating col assignments
            for (cidx = rowIndeces.size(); cidx < indeces.size(); ++cidx)
            {
                if (matasgns[cidx] < matcards[cidx]-1)
                {
                    matasgns[cidx] += 1;
                    iter += matcoeff[cidx];
                    break;
                }
                matasgns[cidx] = 0;
                iter -= matcoeff[cidx]*(matcards[cidx]-1);
            }

        }
        for (ridx = 0; ridx < rowIndeces.size(); ++ridx)
        {
            if (matasgns[ridx] < matcards[ridx]-1)
            {
                matasgns[ridx] += 1;
                iter += matcoeff[ridx];
                break;
            }
            matasgns[ridx] = 0;
            iter -= matcoeff[ridx]*(matcards[ridx]-1);
        }

    }

    return result;
}

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
void
Tensor::fromMat(const cx_mat & matrix,
                const vector<Index> &row, const vector<Index> & col){
    // updating indeces
    indeces.clear();
    indeces.insert(indeces.end(), row.begin(), row.end());
    indeces.insert(indeces.end(), col.begin(), col.end());

    // updating cardinalities
    allCards = prodCards();

    values = vector<cx_d> (allCards, cx_d(0.0,0.0));
    //computing the prod of rows
    vector<int> prodRow(1,1);
    for (int r = 0; r < row.size(); ++r)
        prodRow.push_back(prodRow[r]*row[r].card);

    //computing the prod of cols
    vector<int> prodCol(1,1);
    for (int c = 0; c < col.size(); ++c)
        prodCol.push_back(prodCol[c]*col[c].card);


    int r_temp, c_temp, idx = 0, ridx;

    for (int r = 0; r < prodRow.back(); ++r)
    {
        // finding the corresponding row states
        r_temp = r;
        for (int i = row.size()-1; i > -1; --i)
        {
            idx += coeff[row[i].name] * (r_temp / prodRow[i]);
            r_temp %= prodRow[i];
        }
        ridx = idx;
        for (int c = 0; c < prodCol.back(); ++c)
        {
            // finding the corresponding col states
            c_temp = c;
            for (int i = col.size()-1; i > -1; --i)
            {
                idx += coeff[col[i].name]*(c_temp / prodCol[i]);
                c_temp %= prodCol[i];
            }

            // putting the value into the matrix
            values[idx] = matrix(r,c);
            idx = ridx;
        }
        idx = 0;
    }

}

/**
 * toVec
 * creating a cx_vec from a Tensor
 *
 * return cx_vec the resulting vector
 */
cx_vec
Tensor::toVec (){
    // defining the final vector
    cx_vec result(allCards);
    // putting the values in the vector
    for (int i = 0; i < allCards; ++i) {
        result(i) = values[i];
    }
    return result;
}

/**
 * fromVec
 * creating a tensor from cx_vec vector given the inedeces
 *
 * param vect cx_vec
 * param vecInd
 *
 * return void
 */
void Tensor::fromVec(const arma::cx_vec & vect,
                     const std::vector<Index> & vecInd){
    // updating indeces
    indeces.clear();
    indeces.insert(indeces.end(), vecInd.begin(), vecInd.end());

    // updating cardinalities
    allCards = prodCards();

    values = vector<cx_d> (allCards, cx_d(0.0, 0.0));

    // putting numbers into values
    for (int i = 0 ; i < allCards; ++i){
        values[i] = vect(i);
    }
}

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
vector<vector<Index> >
Tensor::similarities(const Tensor & other){
    // cout << "calling sim" << endl;
    // check whether any of our tensor indeces exist in the other
    vector<vector<Index> > result;
    vector<Index> rowFinal;
    vector<Index> colFinal;
    vector<Index> contracting;
    for (int i = 0; i < indeces.size(); ++i)
    {
        if (!other.coeff.count(indeces[i].name))
            rowFinal.push_back(indeces[i]);
        else
            contracting.push_back(indeces[i]);
    }

    for (int i = 0; i < other.indeces.size(); ++i)
    {
        if (!coeff.count(other.indeces[i].name))
            colFinal.push_back(other.indeces[i]);
    }

    result.push_back(rowFinal);
    result.push_back(colFinal);
    result.push_back(contracting);
    return result;
}

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
Tensor
Tensor::operator * (const Tensor & other){
    // cout << "calling *" << endl;

    //finding the similarities:
    // 3 indeces are needed rowFinal contracting colFinal
    vector<vector<Index> > sims = similarities(other);
    vector<Index> rowFinal = sims[0];
    vector<Index> colFinal = sims[1];
    vector<Index> contracting = sims[2];

    // using the vectors to change the representation of tensors
    prodCards();
    cx_mat one, two;

    cx_mat res = toMat_aux(rowFinal, contracting, one) *
        other.toMat_aux(contracting, colFinal, two);

    // creating a new set of indeces from the result
    vector<Index> indecesFinal (rowFinal);
    for (int i = 0; i < colFinal.size(); ++i)
        indecesFinal.push_back(colFinal[i]);

    // calculating the cardinalities
    int rowCard = 1, colCard = 1;
    for (int r = 0; r < rowFinal.size(); ++r)
        rowCard *=rowFinal[r].card;
    for (int c = 0; c < colFinal.size(); ++c)
        colCard *=colFinal[c].card;
    // cout << rowCard << "\t" << colCard << endl;

    Tensor result(indecesFinal);
    for (int c = 0; c < colCard; ++c)
    {
        for (int r = 0; r < rowCard; ++r)
        {
            result.values.push_back(res(r,c));
        }
    }

    result.prodCards();
    return result;
}

/**
 * overloading operator +
 *
 * the + operator sums the elements of two vector with same indeces
 *
 * return a new Tensor
 */
Tensor
Tensor::operator + (const Tensor & other){
    bool equal= true;
    if (indeces.size() != other.indeces.size())
        equal = false;
    else {
        for (int i = 0; i < indeces.size(); ++i)
        {
            if (indeces[i] == other.indeces[i])
                continue;
            equal = false;
        }
    }

    if (!equal)
        cout << "ERROR: + on not equal indeces Tensors";
    else
    {
        // create a new Tensor
        Tensor res(indeces);
        res.prodCards();
        res.values = vector<cx_d> (values.size(),cx_d(0.0,0.0));
        for (int i = 0; i < values.size(); ++i)
            res.values[i] = values[i] + other.values[i];
        return res;
    }
}

/**
 * overloading operator /
 * divinding all of the elements of the Tensor by a number
 *
 * param num double divisor
 *
 * return reference to the same Tensor
 */
Tensor&
Tensor::operator / (double num){
    for (int i = 0; i < values.size(); ++i)
    {
        values[i] = values[i]/num;
    }
    return *this;
}

/**
 * prodCards
 * finding the cardinality of the tensor and filling the cardinality table
 * in the coeff.
 *
 * return the full cardinality of the Tensor
 */
long
Tensor::prodCards(){
    long prod = 1;
    coeff.clear();
    vecCoeff.clear();
    for (int idx = 0; idx < indeces.size(); ++idx)
    {
        coeff[indeces[idx].name] = prod;
        vecCoeff.push_back(prod);
        prod *= indeces[idx].card;
    }
    allCards = prod;
    return prod;
}

/**
 * conjugate
 * taking the complex conjugate of all the element of the Tensor
 *
 * changes the current Tensor
 * return Tensor conjugated of the same Tensor
 */
Tensor
Tensor::conjugate (){
    double imaginary;
    Tensor result(indeces);
    for (int i = 0; i < values.size(); ++i)
        result.values.push_back(cx_d(values[i].real(), -values[i].imag()));
    return result;
}

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
void
Tensor::reIndex(const Index a1, const Index a2, const Index a3, const Index a4) {
    // calling the original function
    reIndex(mkIdxSet(a1,a2,a3,a4));
}
void
Tensor::reIndex(const vector<Index> & newIndeces){
    // check for sanity of input (TO-DO)
    indeces = newIndeces;
    allCards = prodCards();
}


/**
 * rearrange
 * rearranging the Tensors given new order for the indeces
 * this function just changes the order of elements and indeces
 *
 * return void
 */
void
Tensor::rearrange(const vector<Index> & newOrder){
    // check for sanity of input (TO-DO)
    vector<cx_d> oldvalues = values;
    vector<long> newCards(1,1);
    int idx, ix_temp;
    for (int i = 0; i < newOrder.size(); ++i)
        newCards.push_back(newOrder[i].card*newCards[i]);
    for (int ix = 0; ix < values.size(); ++ix)
    {
        idx = 0;
        ix_temp = ix;
        for (int i = newOrder.size()-1; i > -1; --i)
        {
            idx += (coeff[newOrder[i].name])*(ix_temp/newCards[i]);
            ix_temp %= newCards[i];
        }
        values[ix] = oldvalues[idx];
    }
    indeces = newOrder;
    prodCards();
}

void Tensor::printIndeces() const{
    for (int i = 0; i < indeces.size(); ++i)
        cout << indeces[i].name << ":" <<indeces[i].card << "\t";
    cout << endl;
}

/**
 * mapFinder
 * finding the mapping between indeces with indexes in the full vector
 * param fullIndeces a vector of indexes contating all of the indeces
 *
 * return vector<int> that is the indexes of indeces in the full vector
 */
vector<int>
Tensor::mapFinder(const vector<Index> fullIndeces) const {
    vector<int> idxMap;
    // for (int i = 0; i < fullIndeces.size(); ++i)
    //     {
    //         cout << fullIndeces[i].name << "\t";
    //     }
    // cout << endl;
    for (int i = 0; i < indeces.size(); ++i)
    {
        for (int j = 0; j < fullIndeces.size(); ++j)
        {
            if (indeces[i] == fullIndeces[j])
            {
                idxMap.push_back(j);
                break;
            }
            if (j == fullIndeces.size())
            {
                cout << "ERROR: mapFInder, index on in fullIndeces";
                cout << " : " << indeces[i].name << endl;
            }
        }
    }
    return idxMap;
}

/**
 * getValueOfAsgn
 * getting the value to the given assignment for indeces
 *
 * param asgns is a the given assignments (vector<int>)
 *
 * return complex<double> or cx_d
 */
cx_d Tensor::getValueOfAsgn(const vector<int> asgns) const {
    // finding the index of asgn for values
    int idx = 0;
    for (int i = 0; i < asgns.size(); ++i)
        idx += asgns[i]*vecCoeff[i];
    return values[idx];
}

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
Tensor Tensor::slice(Index index, int from, int upto) {
    Tensor result;
    return result;
}

//  LocalWords:  colFinal rowFinal
