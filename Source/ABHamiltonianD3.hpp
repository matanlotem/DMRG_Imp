/*
 * ABHamiltonianD3.hpp
 *
 *  Created on: Aug 29, 2016
 *      Author: Matan
 */

#ifndef ABHAMILTONIAND3_HPP_
#define ABHAMILTONIAND3_HPP_

#include "ABHamiltonian.hpp"

class ABHamiltonianD3 : public ABHamiltonian {
private:
	MatrixXd HA, HB, SzA, SzB, SminA, SminB, SplusA, SplusB;
	double Jxy, Jz;
	int D;
public:
	ABHamiltonianD3(MatrixXd HA, MatrixXd SzA, MatrixXd SminA, MatrixXd SplusA, double Jxy, double Jz);
	VectorXd apply(VectorXd inputVector);
	VectorXd operator* (VectorXd V);
};

ABHamiltonianD3::ABHamiltonianD3(MatrixXd HA, MatrixXd SzA, MatrixXd SminA, MatrixXd SplusA, double Jxy, double Jz) :
		HA(HA), SzA(SzA), SminA(SminA), SplusA(SplusA), Jxy(Jxy), Jz(Jz) {
	D = HA.rows();
	SzB = SzA.reverse();
	SminB = SminA.reverse();
	SplusB = SplusA.reverse();
	HB = HA.reverse();
}

VectorXd ABHamiltonianD3::apply(VectorXd inputVector) {
	Map<MatrixXd> inputMatrix(inputVector.data(),D,D);
	MatrixXd outputMatrix(D,D);
	VectorXd outputVector(D*D);

	outputMatrix  = HA * (inputMatrix).transpose() + (HB*inputMatrix).transpose();
	outputMatrix += Jxy/2 * SminA * (SplusB * inputMatrix).transpose();
	outputMatrix += Jxy/2 * SplusA * (SminB * inputMatrix).transpose();
	outputMatrix += Jz * SzA * (SzB * inputMatrix).transpose();

	for (int i=0; i<D; i++)
		for (int j=0; j<D; j++)
			outputVector(i*D+j) = outputMatrix(i,j);

	return outputVector;
}

VectorXd ABHamiltonianD3::operator* (VectorXd V) {
	return apply(V);
}

#endif /* ABHAMILTONIAND3_HPP_ */
