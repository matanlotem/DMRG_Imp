/*
 * ABHamiltonian.hpp
 *
 *  Created on: Aug 29, 2016
 *      Author: Matan
 */

#ifndef ABHAMILTONIAN_HPP_
#define ABHAMILTONIAN_HPP_
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "utils.hpp"

using namespace Eigen;
using namespace std;
using utils::tensorProdIdent;

class ABHamiltonian {

public:
	//ABHamiltonian(MatrixXd HA, MatrixXd SzA, MatrixXd SminA, MatrixXd SplusA, double Jxy, double Jz);
	virtual ~ABHamiltonian();

	virtual VectorXd apply(VectorXd inputVector) = 0;
	virtual VectorXd operator* (VectorXd V) = 0;
};

ABHamiltonian::~ABHamiltonian() {}

/*VectorXd ABHamiltonian::apply(VectorXd inputVector) {
	return inputVector;
}

VectorXd ABHamiltonian::operator* (VectorXd V) {
	return apply(V);
}*/

/*VectorXd operator* (ABHamiltonian H, VectorXd V) {
	return H.apply(V);
}*/

#endif /* ABHAMILTONIAN_HPP_ */
