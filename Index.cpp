#include "Index.h"

using namespace std;

void Index::change_card (u_int newCard){
    card = newCard;
}

bool Index::operator == (const Index & other) const {
    return (name == other.name && card == other.card);
}

bool Index::operator < (const Index & other) {
    return (name < other.name);
}

/**
 * mkIdxSet
 * makes a vector of Index out of some number of Indexes
 * overloaded for different number of arguments
 *
 * param Index (some number)
 *
 * return vector<Index>
 */
vector<Index> mkIdxSet (const Index one) {
    vector<Index> idcs;
    idcs.push_back(one);
    return idcs;
}

vector<Index> mkIdxSet (const Index one, const Index two){
    vector<Index> idcs;
    idcs.push_back(one);
    idcs.push_back(two);
    return idcs;
}

vector<Index> mkIdxSet (const Index one, const Index two,
                        const Index three) {
    vector<Index> idcs;
    idcs.push_back(one);
    idcs.push_back(two);
    idcs.push_back(three);
    return idcs;
}

vector<Index> mkIdxSet (const Index one, const Index two,
                        const Index three, const Index four) {
    vector<Index> idcs;
    idcs.push_back(one);
    idcs.push_back(two);
    idcs.push_back(three);
    idcs.push_back(four);
    return idcs;
}
vector<Index> mkIdxSet (const Index one, const Index two,
                        const Index three, const Index four,
                        const Index five, const Index six){
    vector<Index> idcs;
    idcs.push_back(one);
    idcs.push_back(two);
    idcs.push_back(three);
    idcs.push_back(four);
    idcs.push_back(five);
    idcs.push_back(six);
    return idcs;
}
