/**
 * Index header
 */
#ifndef _INDEX_H_
#define _INDEX_H_


#include <string>
#include <vector>
#include <map>

class Index {
 public:
    std::string name;
    long card;

 Index(std::string s, int c) : name(s), card(c){};
    ~Index(){};

    void change_card (int newCard);
    bool operator ==(const Index & other) const;
    bool operator < (const Index & other);
};

class IndexSet {
 public:
    std::vector<Index> idxSet;
    std::map<std::string, long> coeff;

    long prodCards;

    IndexSet (Index one);
    IndexSet (Index one, Index two);
    IndexSet (Index one, Index two, Index three);
    IndexSet (Index one, Index two, Index three, Index four);
    IndexSet (std::vector<Index> & idxs);
    ~IndexSet();

    void createMap();
};

#endif /* _INDEX_H_ */
