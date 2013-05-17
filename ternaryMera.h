#include "Tensor.h"
#include <cstdlib>
#include <ctime>

#ifndef _TERNARYMERA_H_
#define _TERNARYMERA_H_

Tensor ternary_Ascending (Tensor & Hamiltonian,
                          Tensor & Unitary, Tensor & Isometry);

Tensor ternary_Descending (Tensor & DensityMatrix,
                           Tensor & Unitary, Tensor & Isometry);

void ternaryMera (Tensor & Hamiltonian, int max_card);

void giveRandomTensors(Tensor & Unitary, Tensor & Isometry, int in, int out);
void giveRandomDensity(Tensor & DensityMatrix, int card);

double energy(Tensor & Hamiltonian, Tensor & DensityMatrix);

Tensor ternary_Environment_Iso (Tensor & Hamiltonian,
                                Tensor & DensityMatrix,
                                Tensor & Unitary,
                                Tensor & Isometry);

Tensor ternary_Environment_Unit (Tensor & Hamiltonian,
                                 Tensor & DensityMatrix,
                                 Tensor & Unitary,
                                 Tensor & Isometry);


#endif /* _TERNARYMERA_H_ */
