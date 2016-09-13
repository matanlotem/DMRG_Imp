/*
 * BDHamiltonian.hpp
 *
 *  Created on: Sep 4, 2016
 *      Author: Matan
 */

#ifndef BDHAMILTONIAN_HPP_
#define BDHAMILTONIAN_HPP_
#include <Eigen/core>

#include "BDMatrix.hpp"
#include "BODMatrix.hpp"


class BDHamiltonian {
private:
	BDMatrix *HA;
	BDMatrix *SzA;
	BODMatrix *SplusA;
	BDMatrix *HB;
	BDMatrix *SzB;
	BODMatrix *SplusB;
	double Jxy, Jz, Hz, SzTot;
public:
	BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA, double Jxy, double Jz, double Hz);
	BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA, BDMatrix * HB, BDMatrix * SzB, BODMatrix * SplusB, double Jxy, double Jz, double Hz);
	BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA, double Jxy, double Jz, double Hz, double SzTot);
	BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA, BDMatrix * HB, BDMatrix * SzB, BODMatrix * SplusB, double Jxy, double Jz, double Hz, double SzTot);
	~BDHamiltonian();

	void setSzTot(double szTot);
	double getSzTot();

	BDMatrix apply(BDMatrix vector);
	BDMatrix apply(BDMatrix vector, double szTot);
	int dim();
	int dim(double szTot);
	int blockNum();
	int blockNum(double szTot);

	MatrixXd toMatrix();
	void print();

};

BDHamiltonian::BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA, double Jxy, double Jz, double Hz) :
		HA(HA), SzA(SzA), SplusA(SplusA), HB(HA),
		SzB(SzA), SplusB(SplusA), Jxy(Jxy), Jz(Jz), Hz(Hz) {
	SzTot = 0;
}

BDHamiltonian::BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA,
							 BDMatrix * HB, BDMatrix * SzB, BODMatrix * SplusB,
							 double Jxy, double Jz, double Hz) :
		HA(HA), SzA(SzA), SplusA(SplusA), HB(HB),
		SzB(SzB), SplusB(SplusB), Jxy(Jxy), Jz(Jz), Hz(Hz) {
	SzTot = 0;
}

BDHamiltonian::BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA, double Jxy, double Jz, double Hz, double SzTot) :
		HA(HA), SzA(SzA), SplusA(SplusA),
		SzB(SzA), SplusB(SplusA), Jxy(Jxy), Jz(Jz), Hz(Hz), SzTot(SzTot) {
}

BDHamiltonian::BDHamiltonian(BDMatrix * HA, BDMatrix * SzA, BODMatrix * SplusA,
							 BDMatrix * HB, BDMatrix * SzB, BODMatrix * SplusB,
							 double Jxy, double Jz, double Hz, double SzTot) :
		HA(HA), SzA(SzA), SplusA(SplusA),
		HB(HB), SzB(SzB), SplusB(SplusB), Jxy(Jxy), Jz(Jz), Hz(Hz), SzTot(SzTot) {}

BDHamiltonian::~BDHamiltonian() {}

void BDHamiltonian::setSzTot(double szTot) {SzTot = szTot;}
double BDHamiltonian::getSzTot() {return SzTot;}

BDMatrix BDHamiltonian::apply(BDMatrix vector) {
	return apply(vector, SzTot);
}

BDMatrix BDHamiltonian::apply(BDMatrix vector, double szTot) {
	BDMatrix newVector(vector.blockNum());
	int i,I;
	double blockValue;
	for (int blockInd=0; blockInd<vector.blockNum(); blockInd++) {
		blockValue = vector.getBlockValue(blockInd);
		i = HA->getIndexByValue(blockValue);
		I = HB->getIndexByValue(szTot - blockValue);
		if ((i!=-1) && (I!=-1)) {
			//newVector[blockInd] = (*HA)[i] * vector[blockInd] + ((*HA)[I]*(vector[blockInd].transpose())).transpose();
			newVector[blockInd] = (*HA)[i] * vector[blockInd] + vector[blockInd]*(*HB)[I]; // transpose expansion, HA is symmetric

			// newVector[blockInd] += Jz * (*SzA)[i] * ((*SzA)[I] * vector[blockInd].transpose()).transpose();
			// newVector[blockInd] += Jz * (*SzA)[i] * vector[blockInd] * (*SzA)[I]; // transpose expansion, SzA is symmetric
			for (int j=0; j<newVector[blockInd].rows(); j++)
				for (int k=0; k<newVector[blockInd].cols(); k++)
					newVector[blockInd](j,k) += Jz * vector[blockInd](j,k) * (*SzA)[i](j,j) * (*SzB)[I](k,k);

			if (blockInd>0) {
				newVector[blockInd-1] += Jxy/2 * (*SplusA)[i-1] * vector[blockInd] * (*SplusB)[I];
				//newVector[blockInd]   += Jxy/2 * (*SplusA)[i-1].transpose() * ((*SplusA)[I]*vector[blockInd-1].transpose()).transpose();
				newVector[blockInd]   += Jxy/2 * (*SplusA)[i-1].transpose() * vector[blockInd-1] * (*SplusB)[I].transpose(); //transpose expansion
			}
			newVector.setBlockValue(blockInd,blockValue);
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
		I = HB->getIndexByValue(szTot - HA->getBlockValue(i));
		if (I!=-1) d +=  (*HA)[i].rows() * (*HB)[I].rows();
	}
	return d;
}


int BDHamiltonian::blockNum() {
	return blockNum(SzTot);
}

int BDHamiltonian::blockNum(double szTot) {
	int I, b=0;
	for (int i=0; i<HA->blockNum(); i++) {
		I = HB->getIndexByValue(szTot - HA->getBlockValue(i));
		if (I!=-1) b++;
	}
	return b;
}

MatrixXd BDHamiltonian::toMatrix() {
	MatrixXd matrix(dim(),dim());
	BDMatrix vector(blockNum());
	int I, blockInd=0, colInd = 0;
	for (int i=0; i<HA->blockNum(); i++) {
		I = HB->getIndexByValue(0 - HA->getBlockValue(i));
		if (I!=-1) {
			vector[blockInd].resize((*HA)[i].rows(), (*HB)[I].rows());
			vector[blockInd].setZero();
			vector.setBlockValue(blockInd,HA->getBlockValue(i));
			blockInd++;
		}
	}

	for (int b=0; b<vector.blockNum(); b++) {
		for (int i=0; i<vector[b].rows(); i++) {
			for (int j=0; j<vector[b].cols(); j++) {
				vector[b](i,j) = 1;
				matrix.col(colInd++) = apply(vector).flatten().transpose();
				vector[b](i,j) = 0;
			}
		}
	}

	return matrix;
}

void BDHamiltonian::print() {
	cout << toMatrix() << endl;
}




#endif
/* BDHAMILTONIAN_HPP_ */
