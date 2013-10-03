/**
 * @file Index.h
 */
#ifndef _INDEX_H_
#define _INDEX_H_


#include <string>
#include <vector>
#include <map>

typedef unsigned int u_int;

/** class Index
 * a class for representing indeces of a Tensor
 * each Index have string name and a long cardinality
 */
class Index {
public:
    std::string name;
    u_int card;

    Index(std::string s, u_int c) : name(s), card(c){};
    Index(){};
    ~Index(){};

    void i(std::string s, u_int c) {name = s; card = c;};
    void change_card (u_int newCard);
    bool operator ==(const Index & other) const;
    bool operator < (const Index & other);
};

/**
 * mkIdxSet
 * makes a vector of Index out of some number of Indexes
 * overloaded for different number of arguments
 *
 * param Index (some number)
 *
 * return vector<Index>
 */
std::vector<Index> mkIdxSet (const Index one);
std::vector<Index> mkIdxSet (const Index one, const Index two);
std::vector<Index> mkIdxSet (const Index one, const Index two,
                             const Index three);
std::vector<Index> mkIdxSet (const Index one, const Index two,
                             const Index three, const Index four);
std::vector<Index> mkIdxSet (const Index one, const Index two,
                             const Index three, const Index four,
                             const Index five, const Index six);

#endif /* _INDEX_H_ */
