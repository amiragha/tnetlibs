#include "Index.h"

void Index::change_card (int newCard){
    card = newCard;
}

bool Index::operator == (const Index & other) const {
    return (name == other.name);
}

bool Index::operator < (const Index & other) {
    return (name < other.name);
}


IndexSet::IndexSet(Index one){
    idxSet.push_back(one);
    createMap();
}
IndexSet::IndexSet(Index one, Index two){
    idxSet.push_back(one);
    idxSet.push_back(two);
    createMap();
}
IndexSet::IndexSet(Index one, Index two, Index three){
    idxSet.push_back(one);
    idxSet.push_back(two);
    idxSet.push_back(three);
    createMap();

}
IndexSet::IndexSet(Index one, Index two, Index three, Index four){
    idxSet.push_back(one);
    idxSet.push_back(two);
    idxSet.push_back(three);
    idxSet.push_back(four);
    createMap();
}
IndexSet::IndexSet(std::vector<Index> & idxs){
    idxSet = idxs;
    createMap();
}

IndexSet::~IndexSet(){
}

void IndexSet::createMap(){
    long prodC = 1;
    for (int i = 0; i < idxSet.size(); ++i)
        {
            coeff[idxSet[i].name] = prodC;
            prodC *= idxSet[i].card;
        }
    prodCards = prodC;
}
