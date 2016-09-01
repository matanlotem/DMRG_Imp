/*
 * ABHamiltonianFull.hpp
 *
 *  Created on: Aug 29, 2016
 *      Author: Matan
 */

#ifndef ABHAMILTONIANFULL_HPP_
#define ABHAMILTONIANFULL_HPP_

#include "ABHamiltonian.hpp"

class ABHamiltonianFull : public ABHamiltonian {
private:
	MatrixXd HA, HB, SzA, SzB, SminA, SminB, SplusA, SplusB;
	double Jxy, Jz;
	int D;
	MatrixXd HAB;
public:
	ABHamiltonianFull(MatrixXd HA, MatrixXd SzA, MatrixXd SminA, MatrixXd SplusA, double Jxy, double Jz);
	VectorXd apply(VectorXd inputVector);
	VectorXd operator* (VectorXd V);
};

ABHamiltonianFull::ABHamiltonianFull(MatrixXd HA, MatrixXd SzA, MatrixXd SminA, MatrixXd SplusA, double Jxy, double Jz) :
		HA(HA), SzA(SzA), SminA(SminA), SplusA(SplusA), Jxy(Jxy), Jz(Jz) {
	D = HA.rows();
	SzB = SzA.reverse();
	SminB = SminA.reverse();
	SplusB = SplusA.reverse();
	HB = HA.reverse();

	/*HAB = tensorProdIdent(HA,D) + tensorProdIdent(D, HA.reverse())
			+ Jxy/2 * tensorProdIdent(SminA,D)*tensorProdIdent(D,SplusA.reverse())
			+ Jxy/2 * tensorProdIdent(SplusA,D)*tensorProdIdent(D,SminA.reverse())
			+ Jz * tensorProdIdent(SzA,D)*tensorProdIdent(D,SzA.reverse());*/
	HAB = tensorProdIdent(HA,D) + tensorProdIdent(D, HA.reverse());

	for (int i=0; i<D; i+=2) {
		for (int j=0; j<D; j+=2) {
			HAB(i*D+j, i*D+j + D + 1) += Jxy/2;
			HAB(i*D+j + D + 1, i*D+j) += Jxy/2;
		}
	}

	for (int i=0; i<D*D; i++) {
		if (i%2 == (i/D)%2) 	HAB(i,i) -= Jz/4;
		else					HAB(i,i) += Jz/4;
	}

}

VectorXd ABHamiltonianFull::apply(VectorXd inputVector) {
	return HAB*inputVector;
}

VectorXd ABHamiltonianFull::operator* (VectorXd V) {
	return apply(V);
}


#endif /* ABHAMILTONIANFULL_HPP_ */
