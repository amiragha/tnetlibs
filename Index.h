/**
 * @file Index.h
 */
#ifndef _INDEX_H_
#define _INDEX_H_


#include <string>
#include <vector>
#include <map>

/** class Index
 * a class for representing indeces of a Tensor
 * each Index have string name and a long cardinality
 */
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

#endif /* _INDEX_H_ */
