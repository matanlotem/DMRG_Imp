/*
 * BDHamiltonian.hpp
 *
 *  Created on: Sep 4, 2016
 *      Author: Matan
 */

#ifndef BDHAMILTONIAN_HPP_
#define BDHAMILTONIAN_HPP_
#include <Eigen/core>
#include "SBDMatrix.hpp"
#include "SBDODMatrix.hpp"


class BDHamiltonian {
private:
	SBDMatrix *HA;
	SBDMatrix *SzA;
	SBDODMatrix *SplusA;
	double Jxy, Jz, Hz, SzTot;
public:
	BDHamiltonian(SBDMatrix * HA, SBDMatrix * SzA, SBDODMatrix * SplusA, double Jxy, double Jz, double Hz);
	BDHamiltonian(SBDMatrix * HA, SBDMatrix * SzA, SBDODMatrix * SplusA, double Jxy, double Jz, double Hz, double SzTot);
	~BDHamiltonian();

	void setSzTot(double szTot);
	double getSzTot();

	SBDMatrix apply(SBDMatrix vector);
	SBDMatrix apply(SBDMatrix vector, double szTot);
	int dim();
	int dim(double szTot);
	int blockNum();
	int blockNum(double szTot);
};

BDHamiltonian::BDHamiltonian(SBDMatrix * HA, SBDMatrix * SzA, SBDODMatrix * SplusA, double Jxy, double Jz, double Hz) :
		HA(HA), SzA(SzA), SplusA(SplusA), Jxy(Jxy), Jz(Jz), Hz(Hz) {
	SzTot = 0;
}

BDHamiltonian::BDHamiltonian(SBDMatrix * HA, SBDMatrix * SzA, SBDODMatrix * SplusA, double Jxy, double Jz, double Hz, double SzTot) :
		HA(HA), SzA(SzA), SplusA(SplusA), Jxy(Jxy), Jz(Jz), Hz(Hz), SzTot(SzTot){
}

BDHamiltonian::~BDHamiltonian() {
	/*delete HA;
	delete SzA;
	delete SplusA;*/
}

void BDHamiltonian::setSzTot(double szTot) {SzTot = szTot;}
double BDHamiltonian::getSzTot() {return SzTot;}

SBDMatrix BDHamiltonian::apply(SBDMatrix vector) {
	return apply(vector, SzTot);
}

SBDMatrix BDHamiltonian::apply(SBDMatrix vector, double szTot) {
	SBDMatrix newVector(vector.blockNum());
	int I;
	for (int i=0; i<HA->blockNum(); i++) {
		I = HA->getIndexByValue(szTot - HA->getBlockValue(i));
		if (I!=-1) {
			newVector[i] = (*HA)[i] * vector[i] + ((*HA)[I].reverse()*(vector[i].transpose())).transpose();
			for (int j=0; j<newVector[i].rows(); j++)
				for (int k=0; k<newVector[i].cols(); k++)
					newVector[i](j,k) += Jz * vector[i](k,j) * (*SzA)[i](j,j) * (*SzA)[I]((*SzA)[I].rows()-1-k,(*SzA)[I].rows()-1-k);
			if (i>0) {
				newVector[i-1] += Jxy/2 * (*SplusA)[i-1] * (vector[i]*(*SplusA)[I].reverse());
				newVector[i]   += Jxy/2 * (*SplusA)[i-1].transpose() * ((*SplusA)[I].reverse()*vector[i-1].transpose()).transpose();
			}
			//(*HA)[i] * ((*HA)[I].reverse()*vector[i].transpose()).transpose()
		}
	}
	return newVector;
}

int BDHamiltonian::dim() {
	return dim(SzTot);
}

int BDHamiltonian::dim(double szTot) {
	int I, d=0;
	for (int i=0; i<HA->blockNum(); i++) {
		I = HA->getIndexByValue(szTot - HA->getBlockValue(i));
		if (I!=-1) d +=  (*HA)[i].rows() * (*HA)[I].rows();
	}
	return d;
}


int BDHamiltonian::blockNum() {
	return blockNum(SzTot);
}

int BDHamiltonian::blockNum(double szTot) {
	int I, b=0;
	for (int i=0; i<HA->blockNum(); i++) {
		I = HA->getIndexByValue(szTot - HA->getBlockValue(i));
		if (I!=-1) b++;
	}
	return b;
}



#endif /* BDH	AMILTONIAN_HPP_ */
