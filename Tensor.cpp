#include <iostream>
#include "Tensor.h"

using namespace std;
using namespace arma;


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

void Tensor::print(int brk){
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

cx_mat Tensor::toMat (const vector<Index> & rowIndeces,
                      const vector<Index> & colIndeces) const {

    //computing the prod of rows
    vector<int> prodRow(1,1);
    for (int r = 0; r < rowIndeces.size(); ++r)
        prodRow.push_back(prodRow[r]*rowIndeces[r].card);

    //computing the prod of cols
    vector<int> prodCol(1,1);
    for (int c = 0; c < colIndeces.size(); ++c)
        prodCol.push_back(prodCol[c]*colIndeces[c].card);

    cx_mat result = cx_mat(prodRow.back(),
                           prodCol.back() );

    // filling the representation matrix
    int r_temp, c_temp, idx = 0, ridx;

    for (int r = 0; r < prodRow.back(); ++r)
        {
            // finding the corresponding row states
            r_temp = r;
            for (int i = rowIndeces.size()-1; i > -1; --i)
                {
                    idx += coeff.at(rowIndeces[i].name) * (r_temp / prodRow[i]);
                    r_temp %= prodRow[i];
                }
            ridx = idx;
            for (int c = 0; c < prodCol.back(); ++c)
                {
                    // finding the corresponding col states
                    c_temp = c;
                    for (int i = colIndeces.size()-1; i > -1; --i)
                        {
                            idx += coeff.at(colIndeces[i].name)*(c_temp / prodCol[i]);
                            c_temp %= prodCol[i];
                        }

                    // putting the value into the matrix
                    result(r,c) = values[idx];
                    idx = ridx;
                }
            idx = 0;
        }
    return result;
}

void Tensor::fromMat(const cx_mat & matrix,
                     const vector<Index> &row, const vector<Index> & col){
    indeces.clear();
    indeces.insert(indeces.end(), row.begin(), row.end());
    indeces.insert(indeces.end(), col.begin(), col.end());
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

vector<vector<Index> > Tensor::similarities(const Tensor & other){
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

Tensor Tensor::operator * (const Tensor & other){
    // cout << "calling *" << endl;
    //finding the similarities:
    // 3 indeces are needed rowFinal contracting colFinal
    vector<vector<Index> > sims = similarities(other);
    vector<Index> rowFinal = sims[0];
    vector<Index> colFinal = sims[1];
    vector<Index> contracting = sims[2];
    // cout << "rowFinal" << endl;
    // for (int i = 0; i < rowFinal.size(); ++i)
    //     cout << rowFinal[i].name <<"\t";
    // cout << endl;
    // cout << "colFinal" << endl;
    // for (int i = 0; i < colFinal.size(); ++i)
    //     cout << colFinal[i].name <<"\t";
    // cout << endl;
    // cout << "contracting" << endl;
    // for (int i = 0; i < contracting.size(); ++i)
    //     cout << contracting[i].name <<"\t";
    // cout << endl;

    // using the vectors to change the representation of tensors
    prodCards();
    cx_mat res = toMat(rowFinal, contracting) *
        other.toMat(contracting, colFinal);

    // creating a new set of indeces from the result
    vector<Index> indecesFinal (rowFinal);
    for (int i = 0; i < colFinal.size(); ++i)
        indecesFinal.push_back(colFinal[i]);

    // for (int i = 0; i < indecesFinal.size(); ++i)
    //     cout << indecesFinal[i].name << "\t";
    // cout << endl;

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

Tensor Tensor::operator + (const Tensor & other){
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

Tensor& Tensor::operator / (double num){
    for (int i = 0; i < values.size(); ++i)
        {
            values[i] = values[i]/num;
        }
    return *this;
}


long Tensor::prodCards(){
    long prod = 1;
    coeff.clear();
    for (int idx = 0; idx < indeces.size(); ++idx)
        {
            coeff[indeces[idx].name] = prod;
            prod *= indeces[idx].card;
        }
    allCards = prod;
    return prod;
}


void Tensor::conjugate (){
    double imaginary;
    for (int i = 0; i < values.size(); ++i)
        {
            values[i] = cx_d(values[i].real(), -values[i].imag());
            //cout << endl<<i << " :" << values[i];
        }
}

void Tensor::reIndex(const vector<Index> & newIndeces){
    // check for sanity of input (TO-DO)
    indeces = newIndeces;
    coeff.clear();
    allCards = prodCards();
}

void Tensor::rearrange(const vector<Index> & newOrder){
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
//  LocalWords:  colFinal rowFinal
