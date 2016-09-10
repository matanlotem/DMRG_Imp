/*
 * main.cpp
 *
 *  Created on: Aug 24, 2016
 *      Author: Matan
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <vector>
//#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "utils.hpp"
#include "ABHamiltonian.hpp"
#include "ABHamiltonianD3.hpp"
#include "ABHamiltonianFull.hpp"
#include "SBDMatrix.hpp"
#include "SBDODMatrix.hpp"
#include "BDHamiltonian.hpp"

using namespace Eigen;
using namespace std;
using utils::tensorProdIdent;

#define DEBUG = true
#ifdef DEBUG
#define DBG(x) (x)
#else
#define DBG(x)
#endif


/*VectorXd naiveLanczos(MatrixXd matrix) {
	int dim = matrix.cols();*/
VectorXd naiveLanczos(ABHamiltonian *matrix, int dim) {
	int m = min(500, dim);
	double a,b2,norm;
	VectorXd baseState(dim), outputState(dim), prevState(dim), currState(dim), tmpState(dim);
	// generate base state
	baseState.setRandom();
	baseState = baseState / baseState.norm();

	MatrixXd KMatrix(m,m);
	SelfAdjointEigenSolver<MatrixXd> solver, tmpSolver;
	KMatrix.setZero();

	//first iteration
	currState = baseState;
	norm = currState.norm();
	norm = norm*norm; // <u_n|u_n>
	tmpState = *matrix*currState; // H|u_n>

	a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>
	KMatrix(0,0) = a;

	prevState = currState;
	currState = tmpState - a*currState;

	int n=1;
	bool converged = false;
	double currEv, prevEv=0;
	//iterate to find base state
	while (n<m && norm > 0 && !converged) {
		b2 = 1/norm; // 1/<u_n-1|u_n-1>
		norm = currState.norm();
		norm = norm*norm; // <u_n|u_n>
		b2 *= norm; // <u_n|u_n>/<u_n-1|u_n-1>

		tmpState = *matrix*currState; // H|u_n>
		a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>

		KMatrix(n,n) = a;
		KMatrix(n,n-1) = sqrt(b2);
		KMatrix(n-1,n) = sqrt(b2);

		tmpState = tmpState - a*currState - b2*prevState;
		prevState = currState;
		currState = tmpState;

		n++;

		// check convergence
		if (n%10 == 0) {
			tmpSolver.compute(KMatrix.block(0,0,n,n),false);
			currEv= tmpSolver.eigenvalues()[0];
			converged = abs(currEv - prevEv) < 0.00000000000001;
			prevEv = currEv;
		}
	}

	printf("%d iterations\n",n);
	if (n<m) {
		MatrixXd tmpKMatrix = KMatrix.block(0,0,n,n);
		KMatrix = tmpKMatrix;
		m = n;
	}

	//diagonalize
	solver.compute(KMatrix);

	// calculate eigenvector
	VectorXd minEigenVector = solver.eigenvectors().col(0);

	currState = baseState;
	outputState = currState / currState.norm() * minEigenVector(0);

	norm = currState.norm();
	norm = norm*norm; // <u_n|u_n>
	tmpState = *matrix*currState; // H|u_n>
	a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>

	prevState = currState;
	currState = tmpState - a*currState;

	n=1;
	while (n<m) {
		outputState += currState / currState.norm() *minEigenVector(n);;

		b2 = 1/norm; // 1/<u_n-1|u_n-1>
		norm = currState.norm();
		norm = norm*norm; // <u_n|u_n>
		b2 *= norm; // <u_n|u_n>/<u_n-1|u_n-1>

		tmpState = *matrix*currState; // H|u_n>
		a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>

		tmpState = tmpState - a*currState - b2*prevState;
		prevState = currState;
		currState = tmpState;

		n++;
	}

	return outputState;
}

void tryDMRG(int N) {
	double Jxy=1, Jz=1, Hz=0;

	MatrixXd SzA1(2,2), SminA1(2,2), SplusA1(2,2),
			 SzAPrev, SminAPrev, SplusAPrev, HAPrev,
			 SzACurr, SminACurr, SplusACurr, HA,
			 densityMatrix, transNew2Old, transOld2New;
	VectorXd ABBaseState;
	ABHamiltonian * HAB;
	SelfAdjointEigenSolver<MatrixXd> densitySolver;

	int D = 2, n = 2, chi = 32;

	SzA1 << -0.5,0,
			 0,0.5;
	SminA1 << 0,1,
			  0,0;
	SplusA1 << 0,0,
			   1,0;

	SzAPrev = SzA1;
	SminAPrev = SminA1;
	SplusAPrev = SplusA1;
	HAPrev = Hz*SzAPrev;

	while (n<N) {
		n += 2;
		DBG(printf("n=%d D=%d, HAPrev.dim=%dx%d\n",n,D,(int) HAPrev.rows(),(int) HAPrev.cols()));

		// Add site - enlarge basis
		DBG(printf("\tenlarging basis\n"));
		SzAPrev = tensorProdIdent(SzAPrev,2);
		SminAPrev= tensorProdIdent(SminAPrev,2);
		SplusAPrev= tensorProdIdent(SplusAPrev,2);
		HAPrev = tensorProdIdent(HAPrev,2);

		// create new site operators
		DBG(printf("\tcreating new site operators\n"));
		SzACurr = tensorProdIdent(D,SzA1);
		SminACurr = tensorProdIdent(D,SminA1);
		SplusACurr = tensorProdIdent(D,SplusA1);
		// calculate new Hamiltonian for A
		DBG(printf("\tcalculating new Hamiltonian for A\n"));
		HA = HAPrev + Jxy/2 * (SminAPrev*SplusACurr + SplusAPrev*SminACurr) + Jz * SzAPrev * SzACurr + Hz * SzACurr;
		D = D * 2;

		if (D > chi) {
			// calculate Hamiltonian for AB
			DBG(printf("\tcalculating new Hamiltonian for AB\n"));

			/*HAB = tensorProdIdent(HA,D) + tensorProdIdent(D, HA.reverse());

			for (int i=0; i<D; i+=2) {
				for (int j=0; j<D; j+=2) {
					HAB(i*D+j, i*D+j + D + 1) += Jxy/2;
					HAB(i*D+j + D + 1, i*D+j) += Jxy/2;
				}
			}

			for (int i=0; i<D*D; i++) {
				if (i%2 == (i/D)%2) 	HAB(i,i) -= Jz/4;
				else					HAB(i,i) += Jz/4;
			}*/

			/*HAB2 = tensorProdIdent(HA,D) + tensorProdIdent(D, HA.reverse())
					+ Jxy/2 * tensorProdIdent(SminACurr,D)*tensorProdIdent(D,SplusACurr.reverse())
					+ Jxy/2 * tensorProdIdent(SplusACurr,D)*tensorProdIdent(D,SminACurr.reverse())
					+ Jz * tensorProdIdent(SzACurr,D)*tensorProdIdent(D,SzACurr.reverse());

			for (int i=0; i<D*D; i++) {
				for(int j=0; j<D*D; j++) {
					if (abs(HAB(i,j) - HAB2(i,j)) > 0.0000001) {
						printf("different (%d,%d)\n", i, j);
						cout << HAB2.block(i,j,10,10) << endl << endl << HAB.block(i,j,10,10) << endl << endl;
						abort();
					}
				}
			}*/
			//HAB = new ABHamiltonianD3(HA,SzACurr,SminACurr,SplusACurr, Jxy,Jz);
			HAB = new ABHamiltonianFull(HA,SzACurr,SminACurr,SplusACurr, Jxy,Jz);

			// find AB base state using Lanczos
			DBG(printf("\tfinding AB base state with Lanczos\n"));
			ABBaseState = naiveLanczos(HAB, D*D);
			DBG(printf("\tn=%d min eigenvalue: %.15f\n",n, (*HAB*ABBaseState).norm()));
			delete HAB;

			// create density operator
			DBG(printf("\tcalculating density operator\n"));
			densityMatrix.resize(D,D);
			densityMatrix.setZero();
			for (int i=0; i<D; i++) {
				for (int I=i; I<D; I++) {
					for (int j=0; j<D; j++)
						densityMatrix(i,I) += ABBaseState(i*D+j)*ABBaseState(I*D+j);
					densityMatrix(I,i) = densityMatrix(i,I);
				}
			}

			// diagonalize density operator and truncate basis
			DBG(printf("\tdiagnolaizing density operator\n"));
			densitySolver.compute(densityMatrix);
			transNew2Old = densitySolver.eigenvectors().block(0,D-chi,D,chi);
			transOld2New = transNew2Old.transpose();

			// transform operators to new basis
			DBG(printf("\ttransforming operators to new basis\n"));
			SzAPrev = transOld2New * SzACurr * transNew2Old;
			SminAPrev = transOld2New * SminACurr * transNew2Old;
			SplusAPrev = transOld2New * SplusACurr * transNew2Old;
			HAPrev = transOld2New * HA * transNew2Old;
			D = chi;
		}
		else {
			SzAPrev = SzACurr;
			SminAPrev = SminACurr;
			SplusAPrev = SplusACurr;
			HAPrev = HA;
		}
	}
	DBG(printf("Done: n=%d D=%d\n\n",n,D));
}

void newDMRG(int N) {
	double Jxy=1, Jz=1, Hz=0;
	SelfAdjointEigenSolver<MatrixXd> solver;

	MatrixXd *HAPrevBlocks, *SzAPrevBlocks, *SplusAPrevBlocks,
			 *HACurrBlocks, *SzACurrBlocks, *SplusACurrBlocks;
	double *SzPrevTot, *SzCurrTot;

	int n=1, b = 2, dim = 2, dimMax = 2, D=8;
	int dim1, dim2, dim3;
	MatrixXd HAMatrix;

	DBG(printf("Creating initial matrices\n"));
	HAPrevBlocks = new MatrixXd[b];
	SzAPrevBlocks = new MatrixXd[b];
	SplusAPrevBlocks = new MatrixXd[b-1];
	SzAPrevBlocks[0] = MatrixXd(1,1);
	SzAPrevBlocks[0] << -0.5;
	HAPrevBlocks[0] = Hz * SzAPrevBlocks[0];
	SzAPrevBlocks[1] = MatrixXd(1,1);
	SzAPrevBlocks[1] << 0.5;
	HAPrevBlocks[1] = Hz * SzAPrevBlocks[1];
	SplusAPrevBlocks[0] = MatrixXd(1,1);
	SplusAPrevBlocks[0] << 1;
	SzPrevTot = new double[b];
	for (int i=0; i<b; i++) SzPrevTot[i] = i - double(b)/2;


	while (n<N) {
		DBG(printf("Adding site %d\n", n));
		b++;
		HACurrBlocks = new MatrixXd[b];
		SzACurrBlocks = new MatrixXd[b];
		SplusACurrBlocks = new MatrixXd[b-1];
		SzCurrTot = new double[b];

		DBG(printf("Creating new edge blocks\n"));
		HACurrBlocks[0] = HAPrevBlocks[0] + Jz * -0.5 * SzAPrevBlocks[0];
		SzACurrBlocks[0] = -0.5*MatrixXd::Identity(HACurrBlocks[0].rows(), HACurrBlocks[0].cols());
		HACurrBlocks[b-1] = HAPrevBlocks[b-2] + Jz * 0.5 * SzAPrevBlocks[b-2];
		SzACurrBlocks[b-1] = 0.5*MatrixXd::Identity(HACurrBlocks[b-1].rows(), HACurrBlocks[b-1].cols());
		SzCurrTot[0] = SzPrevTot[0] - 0.5;
		SzCurrTot[b-1] = SzPrevTot[b-2] + 0.5;

		dim1 = HAPrevBlocks[b-3].rows();
		dim2 = HAPrevBlocks[b-2].rows();
		SplusACurrBlocks[b-2] = MatrixXd(dim1+dim2,dim2);
		SplusACurrBlocks[b-2].setZero();
		for (int j=0; j<dim2;j++) SplusACurrBlocks[b-2](dim1+j,j) = 1;

		dim = HACurrBlocks[0].rows() + HACurrBlocks[b-1].rows();
		dimMax = max(HACurrBlocks[0].rows(),HACurrBlocks[b-1].rows());


		DBG(printf("Creating new central blocks\n"));
		for (int i=1; i<b-1; i++) {
			DBG(printf("\tblock %d\n",i));
			dim1 = HAPrevBlocks[i-1].rows();
			dim2 = HAPrevBlocks[i].rows();
			dim += dim1 + dim2;
			HACurrBlocks[i] = MatrixXd(dim1+dim2,dim1+dim2);
			SzACurrBlocks[i] = MatrixXd(dim1+dim2,dim1+dim2);
			SzACurrBlocks[i].setZero();

			// add spin down block
			DBG(printf("\tadding spin down block\n"));
			for (int j=0; j<dim1; j++) SzACurrBlocks[i](j,j) = 0.5;
			HACurrBlocks[i].block(0,0,dim1,dim1) = HAPrevBlocks[i-1] +
					Jz * SzAPrevBlocks[i-1] * SzACurrBlocks[i].block(0,0,dim1,dim1);
			for (int j=0; j<dim1; j++) HACurrBlocks[i](j,j) += 0.5*Hz;

			// add spin up block
			DBG(printf("\tadding spin up block\n"));
			for (int j=0; j<dim2; j++) SzACurrBlocks[i](dim1+j,dim1+j) = -0.5;
			HACurrBlocks[i].block(dim1,dim1,dim2,dim2) = HAPrevBlocks[i] +
						Jz * SzAPrevBlocks[i] * SzACurrBlocks[i].block(dim1,dim1,dim2,dim2);
			for (int j=0; j<dim2; j++) HACurrBlocks[i](dim1+j,dim1+j) += -0.5*Hz;


			// add old block movers Jxy/2 * (S+- + S-+)
			DBG(printf("\tadding Jxy*(S+- + S-+) block\n"));
			HACurrBlocks[i].block(0,dim1,dim1,dim2) = Jxy/2 * SplusAPrevBlocks[i-1];
			HACurrBlocks[i].block(dim1,0,dim2,dim1) = Jxy/2 * SplusAPrevBlocks[i-1].transpose();

			//create new block movers (S+)
			dim3 = HACurrBlocks[i-1].rows();
			DBG(printf("\tcreating new S+\n"));
			SplusACurrBlocks[i-1] = MatrixXd(dim3,dim1+dim2);
			SplusACurrBlocks[i-1].setZero();
			for (int j=0; j<dim1;j++) SplusACurrBlocks[i-1](dim3-dim1+j,j) = 1;

			SzCurrTot[i] = SzPrevTot[i-1] + 0.5;
		}

		// printing stuff
		DBG(printf("printing HA\n"));
		HAMatrix.resize(dim,dim);
		HAMatrix.setZero();
		int ind = 0;
		for (int i=0; i<b; i++) {
			dim1 = HACurrBlocks[i].rows();
			HAMatrix.block(ind,ind,dim1,dim1) = HACurrBlocks[i];
			ind += dim1;
		}
		cout << HAMatrix << endl << endl;
		solver.compute(HAMatrix,false);
		cout << solver.eigenvalues().transpose() << endl << endl;

		// cleaning up and preparing next iteration
		delete[] HAPrevBlocks;
		delete[] SzAPrevBlocks;
		delete[] SplusAPrevBlocks;
		delete SzPrevTot;

		if (dimMax <= D) {
			HAPrevBlocks = HACurrBlocks;
			SzAPrevBlocks = SzACurrBlocks;
			SplusAPrevBlocks = SplusACurrBlocks;
			SzPrevTot = SzCurrTot;
		}
		else { //truncate
			// for each total Sz
				// prepare AB Hamiltonian

				// find AB base state with Lanczos

			// choose AB base state

			// create density marix

			// diagonalize density matrix

			// transform operators to new basis
		}

		n++;
	}
}

SBDMatrix oopLanczos(BDHamiltonian *matrix, SBDMatrix baseState) {
	int m = min(500, matrix->dim());
	double a,b2,norm;
	double tol = 0.0000001;

	SBDMatrix prevState(baseState.blockNum()),
			  currState(baseState.blockNum()),
			  tmpState(baseState.blockNum()),
			  outputState(baseState.blockNum());

	MatrixXd KMatrix(m,m);
	SelfAdjointEigenSolver<MatrixXd> solver, tmpSolver;
	KMatrix.setZero();

	//first iteration
	currState = baseState;

	norm = currState.norm();
	norm = norm*norm; // <u_n|u_n>
	//printf("AAA\n");
	tmpState = matrix->apply(currState); // H|u_n>
	//printf("BBB\n");
	a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>
	KMatrix(0,0) = a;

	prevState = currState;
	//currState = tmpState - a*currState;
	for (int b=0; b<currState.blockNum(); b++) currState[b] = tmpState[b] - a*currState[b];
	//printf("CCC\n");
	int n=1;
	bool converged = false;
	double currEv, prevEv=0;
	//iterate to find base state
	while (n<m && norm > 0 && !converged) {
		b2 = 1/norm; // 1/<u_n-1|u_n-1>
		norm = currState.norm();
		norm = norm*norm; // <u_n|u_n>
		b2 *= norm; // <u_n|u_n>/<u_n-1|u_n-1>
		tmpState = matrix->apply(currState); // H|u_n>
		a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>
		KMatrix(n,n) = a;
		KMatrix(n,n-1) = sqrt(b2);
		KMatrix(n-1,n) = sqrt(b2);

		//tmpState = tmpState - a*currState - b2*prevState;
		for (int b=0; b<currState.blockNum(); b++) tmpState[b] = tmpState[b] - a*currState[b] - b2*prevState[b];
		prevState = currState;
		currState = tmpState;

		n++;

		// check convergence
		if (n%10 == 0) {
			tmpSolver.compute(KMatrix.block(0,0,n,n),false);
			currEv= tmpSolver.eigenvalues()[0];
			converged = abs(currEv - prevEv) < tol;
			prevEv = currEv;
		}
	}

	printf("%d iterations\n",n);
	if (n<m) {
		MatrixXd tmpKMatrix = KMatrix.block(0,0,n,n);
		KMatrix = tmpKMatrix;
		m = n;
	}

	//diagonalize
	solver.compute(KMatrix);

	// calculate eigenvector
	VectorXd minEigenVector = solver.eigenvectors().col(0);

	currState = baseState;

	//outputState = currState / currState.norm() * minEigenVector(0);
	for (int b=0; b<currState.blockNum(); b++) outputState[b] = currState[b] / currState.norm() * minEigenVector(0);

	norm = currState.norm();
	norm = norm*norm; // <u_n|u_n>
	tmpState = matrix->apply(currState); // H|u_n>
	a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>

	prevState = currState;
	//currState = tmpState - a*currState;
	for (int b=0; b<currState.blockNum(); b++) currState[b] = tmpState[b] - a*currState[b];

	n=1;
	while (n<m) {
		//outputState += currState / currState.norm() *minEigenVector(n);;
		for (int b=0; b<currState.blockNum(); b++) outputState[b] += currState[b] / currState.norm() * minEigenVector(n);

		b2 = 1/norm; // 1/<u_n-1|u_n-1>
		norm = currState.norm();
		norm = norm*norm; // <u_n|u_n>
		b2 *= norm; // <u_n|u_n>/<u_n-1|u_n-1>

		tmpState = matrix->apply(currState); // H|u_n>
		a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>


		//tmpState = tmpState - a*currState - b2*prevState;
		for (int b=0; b<currState.blockNum(); b++) tmpState[b] = tmpState[b] - a*currState[b] - b2*prevState[b];
		prevState = currState;
		currState = tmpState;

		n++;
	}

	for (int i=0; i<baseState.blockNum(); i++) outputState.setBlockValue(i, baseState.getBlockValue(i));
	//printf("DDD\n");
	return outputState;
}


void oopDMRG(int N) {
	double Jxy=1, Jz=1, Hz=0;

	int n=1, b=2, D=8;
	SBDMatrix HAPrev(b), SzAPrev(b),
			  HACurr(b), SzACurr(b),
			  DensityMatrix(b);
	SBDODMatrix SplusAPrev(b-1), SplusACurr(b-1);
	BDHamiltonian * HAB;
	SelfAdjointEigenSolver<MatrixXd> solver;
	vector<SelfAdjointEigenSolver<MatrixXd> > blockSolvers;
	vector<int> newBasisVectors;


	//SzAPrev = new SBDMatrix(b);
	SzAPrev[0] = (MatrixXd::Identity(1,1) * -0.5);
	SzAPrev[1] = (MatrixXd::Identity(1,1) *  0.5);

	for (int i=0; i<b; i++) {
		HAPrev[i]=Hz*SzAPrev[i];
		HAPrev.setBlockValue(i,SzAPrev[i](0,0));
		SzAPrev.setBlockValue(i,SzAPrev[i](0,0));
	}

	SplusAPrev.resize(&HAPrev);
	SplusAPrev[0] = MatrixXd::Identity(1,1);

	while (n<N) {
		DBG(printf("Adding site %d\n", n));
		// count new block number
		for (int i=1; i<HAPrev.blockNum(); i++)
			if (HAPrev.getBlockValue(i) - 1 != HAPrev.getBlockValue(i-1)) b++;
		b++;

		DBG(printf("Creating new blocks\n"));
		HACurr.resize(b);
		SzACurr.resize(b);

		int dim1, dim2, blockInd = 0;
		for (int i=0; i<HAPrev.blockNum(); i++) {
			// first block if no block with SzTot-1 exists
			if ((i==0) || (HAPrev.getBlockValue(i) - 1 != HAPrev.getBlockValue(i-1))) {
				HACurr.setBlockValue(blockInd,HAPrev.getBlockValue(i)-0.5);
				dim1 = HAPrev[i].rows();

				DBG(printf("\tadding spin down block\n"));
				SzACurr[blockInd] = MatrixXd::Identity(dim1,dim1) * -0.5;
				HACurr[blockInd] = HAPrev[i] + Jz * SzAPrev[i] * SzACurr[blockInd];
				for (int j=0; j<dim1; j++) HACurr[blockInd](j,j) += -0.5*Hz;
			}
			// if block with SzTot-1 exists
			else {
				HACurr.setBlockValue(blockInd,HAPrev.getBlockValue(i)-0.5);
				//DBG(printf("\tblock %d\n",i));
				dim1 = HAPrev[i-1].rows();
				dim2 = HAPrev[i].rows();

				HACurr[blockInd] = MatrixXd(dim1+dim2,dim1+dim2);
				SzACurr[blockInd] = MatrixXd(dim1+dim2,dim1+dim2);
				SzACurr[blockInd].setZero();

				// add spin up block
				DBG(printf("\tadding spin up block\n"));
				for (int j=0; j<dim1; j++) SzACurr[i](j,j) = 0.5;
				HACurr[blockInd].topLeftCorner(dim1,dim1) = HAPrev[i-1] +
							Jz * SzAPrev[i-1] * SzACurr[blockInd].topLeftCorner(dim1,dim1);
				for (int j=0; j<dim1; j++) HACurr[blockInd](j,j) += 0.5*Hz;

				// add spin down block
				DBG(printf("\tadding spin down block\n"));
				for (int j=dim1; j<dim1+dim2; j++) SzACurr[blockInd](j,j) = -0.5;
				HACurr[blockInd].bottomRightCorner(dim2,dim2) = HAPrev[i] +
							Jz * SzAPrev[i] * SzACurr[blockInd].bottomRightCorner(dim2,dim2);
				for (int j=dim1; j<dim1+dim2; j++) HACurr[blockInd](j,j) += -0.5*Hz;

				// add old block movers Jxy/2 * (S+- + S-+)
				DBG(printf("\tadding Jxy*(S+- + S-+) block\n"));
				HACurr[blockInd].topRightCorner(dim1,dim2) = Jxy/2 * SplusAPrev[i-1];
				HACurr[blockInd].bottomLeftCorner(dim2,dim1) = Jxy/2 * SplusAPrev[i-1].transpose();
			}
			blockInd++;

			// if last block or no block with SzTot+1 exists
			if ((i==HAPrev.blockNum() - 1) || (HAPrev.getBlockValue(i) + 1 != HAPrev.getBlockValue(i+1))) {
				HACurr.setBlockValue(blockInd,HAPrev.getBlockValue(i)+0.5);
				dim1 = HAPrev[i].rows();

				DBG(printf("\tadding spin up block\n"));
				SzACurr[blockInd] = MatrixXd::Identity(dim1,dim1) * 0.5;
				HACurr[blockInd] = HAPrev[i] + Jz * SzAPrev[i] * SzACurr[blockInd];
				for (int j=0; j<dim1; j++) HACurr[blockInd](j,j) += 0.5*Hz;

				blockInd++;
			}

		}

		//create new block movers (S+)
		DBG(printf("creating new S+ blocks\n"));
		SplusACurr.resize(&HACurr);
		for (int i=0; i<SplusACurr.blockNum(); i++) {
			if (HACurr.getBlockValue(i)+1 == HACurr.getBlockValue(i+1)) {
				dim1 = HAPrev[HAPrev.getIndexByValue(HACurr.getBlockValue(i)+0.5)].rows();
				dim2 = HACurr[i].rows();

				if ((dim1==dim2) && (dim1 == HACurr[i+1].cols()))
					SplusACurr[i] = MatrixXd::Identity(dim1,dim1);
				else {
					SplusACurr[i].setZero();
					for (int j=0; j<dim1;j++) SplusACurr[i](dim2-dim1+j,j) = 1;
				}
			}
			else // if no block with SzTot+1 exists
				SplusACurr[i].setZero();

		}


		// printing stuff
		//DBG(printf("printing HA\n"));
		//HACurr.print();
		//solver.compute(HACurr.toMatrix(),false);
		//cout << solver.eigenvalues().transpose() << endl << endl;

		if (HACurr.rows() <= D) {
			SzAPrev = SzACurr;
			HAPrev = HACurr;
			SplusAPrev = SplusACurr;
		}

		else { //truncate
			DBG(printf("creating AB Hamiltonian\n"));
			HAB = new BDHamiltonian(&HACurr, &SzACurr, &SplusACurr, Jxy, Jz, Hz);


			// create base state
			DBG(printf("creating AB initial base state\n"));
			VectorXd baseState(HAB->dim());
			baseState.setRandom();
			baseState = baseState / baseState.norm();
			SBDMatrix baseStateMatrix(HAB->blockNum());
			int I, vecInd=0,blockInd=0;
			for (int i=0; i<HACurr.blockNum(); i++) {
				I = HACurr.getIndexByValue(0 - HACurr.getBlockValue(i));
				if (I!=-1) {
					baseStateMatrix[blockInd].resize(HACurr[i].rows(), HACurr[I].rows());
					for (int j=0; j<HACurr[i].rows(); j++)
						for (int k=0; k<HACurr[I].rows(); k++)
							baseStateMatrix[blockInd](j,k) = baseState(vecInd++);
					baseStateMatrix.setBlockValue(blockInd, HACurr.getBlockValue(i));
					blockInd++;
				}
			}



			// find AB base state with Lanczos

			//HACurr.print();
			//SzACurr.print();
			//SplusACurr.printFullStats();
			//cout << SplusACurr[0] << endl << endl << SplusACurr[1] << endl << endl;
			//HAB->print();
			/*
			printf("aabbcc\n");
			printf("HA:\n");
			HACurr.printFullStats();
			printf("SzA:\n");
			HACurr.printFullStats();
			printf("base state:\n");
			baseStateMatrix.printFullStats();
			baseStateMatrix.print();
			(HAB->apply(baseStateMatrix)).printFullStats();
			(HAB->apply(baseStateMatrix)).print();
			printf("ccbbaa\n");*/
			DBG(printf("finding AB base state with Lanczos\n"));
			SBDMatrix ABBaseState = oopLanczos(HAB, baseStateMatrix);
			DBG(printf("base Ev: %f\n",HAB->apply(ABBaseState).norm()));
			//ABBaseState.print();

			// create density matrix
			DBG(printf("creating A density matrix\n"));
			DensityMatrix.resize(HAB->blockNum());
			for (int i=0; i<DensityMatrix.blockNum(); i++) {
				DensityMatrix[i] = ABBaseState[i].transpose()*ABBaseState[i];
				DensityMatrix.setBlockValue(i, ABBaseState.getBlockValue(i));
			}
			//DensityMatrix.print();

			// diagonalize density matrix
			DBG(printf("diagonalizing A density matrix\n"));
			blockSolvers.resize(DensityMatrix.blockNum());
			for (int i=0; i<DensityMatrix.blockNum(); i++)
				blockSolvers[i].compute(DensityMatrix[i]);

			//solver.compute(DensityMatrix.toMatrix(),false);
			//cout << "DMatrix ev: " << solver.eigenvalues().transpose() << endl << endl;

			// select new basis vectors
			DBG(printf("selecting new basis vectors\n"));
			newBasisVectors.resize(DensityMatrix.blockNum());
			for (int i=0; i<(int) newBasisVectors.size(); i++) newBasisVectors[i] = 0;

			int vectorNum = 0, currBlock = 0;
			double maxEv;
			while (vectorNum < D) {
				maxEv = 0;
				currBlock = 0;
				for (int i=0; i<(int) newBasisVectors.size(); i++) {
					if (newBasisVectors[i] < DensityMatrix[i].rows()) {
						if (blockSolvers[i].eigenvalues()[DensityMatrix[i].rows() - 1 - newBasisVectors[i]] > maxEv) {
							maxEv = blockSolvers[i].eigenvalues()[DensityMatrix[i].rows() - 1 - newBasisVectors[i]];
							currBlock = i;
						}
					}
				}
				/*printf("block=%d ev=%f\n",currBlock,maxEv);
				printf("\tev check: %f\n", (DensityMatrix[currBlock] *
						blockSolvers[currBlock].eigenvectors().col(DensityMatrix[currBlock].cols() - 1 - newBasisVectors[currBlock])).norm());
						*/
				newBasisVectors[currBlock]++;
				vectorNum++;
			}


			// count truncated basis blocks;
			b = 0;
			for (int i=0; i<(int) newBasisVectors.size(); i++)
				if (newBasisVectors[i] > 0) b++;

			// transform operators to new basis
			DBG(printf("transforming matrices to new basis\n"));
			SzAPrev.resize(b);
			HAPrev.resize(b);
			SplusAPrev.resize(b-1);

			blockInd = 0;

			for (int i=0; i < (int) newBasisVectors.size(); i++) {
				if (newBasisVectors[i] > 0) {
					//cout << "Density Matrix Sz=" << DensityMatrix.getBlockValue(i) << " EV:\n  " << blockSolvers[i].eigenvalues().bottomRows(newBasisVectors[i]).transpose() << endl;
					SzAPrev[blockInd] = blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]).transpose() *
							            SzACurr[i] * blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]);
					SzAPrev.setBlockValue(blockInd, DensityMatrix.getBlockValue(i));

					HAPrev[blockInd] = blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]).transpose() *
									   HACurr[i] * blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]);
					HAPrev.setBlockValue(blockInd, DensityMatrix.getBlockValue(i));

					if (blockInd>0)
						SplusAPrev[blockInd-1] = blockSolvers[i-1].eigenvectors().rightCols(newBasisVectors[i-1]).transpose() *
											     SplusACurr[i-1] * blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]);

					blockInd++;
				}
			}

			delete HAB;

			//HAPrev.print();
			//solver.compute(HAPrev.toMatrix(),false);
			//cout << endl << solver.eigenvalues().transpose() << endl << endl;
			//printf("new b=%d!\n",b);
			//abort();

			//SzAPrev = SzACurr;
			//HAPrev = HACurr;
			//SplusAPrev = SplusACurr;
		}

		/*solver.compute(HACurr.toMatrix(),false);
		printf("\nHACurr: min ev=%f\n",solver.eigenvalues()[0]);
		HACurr.printFullStats();
		HACurr.print();
		solver.compute(HAPrev.toMatrix(),false);
		printf("\nHAPrev: min ev=%f\n",solver.eigenvalues()[0]);
		HAPrev.printFullStats();
		HAPrev.print();*/

		n += 2;
	}




	printf("yay!\n");

}

void testHAB(int N) {
	double Jxy=11, Jz=13, Hz=0;

	MatrixXd SzA1(2,2), SminA1(2,2), SplusA1(2,2),
			 SzAPrev, SminAPrev, SplusAPrev, HAPrev,
			 SzACurr, SminACurr, SplusACurr, HA,
			 densityMatrix, transNew2Old, transOld2New;

	VectorXd ABBaseState;
	SelfAdjointEigenSolver<MatrixXd> densitySolver;

	int D = 2, n = 2;

	SzA1 << -0.5,0,
			 0,0.5;
	SminA1 << 0,1,
			  0,0;
	SplusA1 << 0,0,
			   1,0;

	SzAPrev = SzA1;
	SminAPrev = SminA1;
	SplusAPrev = SplusA1;
	HAPrev = Hz*SzAPrev;

	while (n<N) {
		n += 2;
		DBG(printf("n=%d D=%d, HAPrev.dim=%dx%d\n",n,D,(int) HAPrev.rows(),(int) HAPrev.cols()));

		// Add site - enlarge basis
		DBG(printf("\tenlarging basis\n"));
		SzAPrev = tensorProdIdent(SzAPrev,2);
		SminAPrev= tensorProdIdent(SminAPrev,2);
		SplusAPrev= tensorProdIdent(SplusAPrev,2);
		HAPrev = tensorProdIdent(HAPrev,2);

		// create new site operators
		DBG(printf("\tcreating new site operators\n"));
		SzACurr = tensorProdIdent(D,SzA1);
		SminACurr = tensorProdIdent(D,SminA1);
		SplusACurr = tensorProdIdent(D,SplusA1);
		// calculate new Hamiltonian for A
		DBG(printf("\tcalculating new Hamiltonian for A\n"));
		HA = HAPrev + Jxy/2 * (SminAPrev*SplusACurr + SplusAPrev*SminACurr) + Jz * SzAPrev * SzACurr + Hz * SzACurr;
		D = D * 2;

		SzAPrev = SzACurr;
		SminAPrev = SminACurr;
		SplusAPrev = SplusACurr;
		HAPrev = HA;
	}

	//cout << HA << endl << endl;
	//clock_t t0,t1,t2,t3,t4;
	//t0=clock();
	//ABHamiltonian HAB(HA,SzACurr,SminACurr,SplusACurr, Jxy,Jz);
	//ABHamiltonianD3 HAB(HA,SzACurr,SminACurr,SplusACurr, Jxy,Jz);
	//ABHamiltonianFull HAB2(HA,SzACurr,SminACurr,SplusACurr, Jxy,Jz);

	VectorXd baseVector(D*D);
	baseVector.setRandom();
	baseVector = baseVector / baseVector.norm();
	cout << baseVector.transpose() << endl << endl;
	ABHamiltonian * HAB;
	HAB = new ABHamiltonianD3(HA,SzACurr,SminACurr,SplusACurr, Jxy,Jz);
	cout << (*HAB *baseVector).transpose() << endl << endl;
	delete HAB;
	HAB = new ABHamiltonianFull(HA,SzACurr,SminACurr,SplusACurr, Jxy,Jz);
	cout << (*HAB * baseVector).transpose() << endl << endl;
	delete HAB;

	/*//baseVector.setZero();
	//baseVector(0) = 1;
	t1=clock();
	for (int i=0; i<1000; i++) {
		baseVector.setRandom();
		baseVector = baseVector / baseVector.norm();
	}
	t2=clock();
	for (int i=0; i<1000; i++) {
		baseVector.setRandom();
		baseVector = baseVector / baseVector.norm();
		HAB.apply(baseVector);
	}
	t3=clock();
	for (int i=0; i<1000; i++) {
		baseVector.setRandom();
		baseVector = baseVector / baseVector.norm();
		//HAB.apply2(baseVector);
	}
	t4=clock();
	printf("clean: %f seconds\n", double(t2-t1)/CLOCKS_PER_SEC);
	printf("new: %f seconds\n", double(t3-t2)/CLOCKS_PER_SEC);
	printf("old: %f seconds\n", double(t4-t3)/CLOCKS_PER_SEC);*/




}

void memoryTests(int N) {
	int d = (1<<N);
	int x;
	MatrixXd m,tmp;
	m.resize(d,d);
	m.setZero();
	tmp.resize(d,d);
	for (int i=0; i<6; i++) {
		cout << "bla: ";
		cin >> x;
		tmp.resize(d,d);
		std::srand(i);
		tmp.setRandom();
		m = 2*m + tmp + tmp;
		m = tmp;
	}

}



int main(int argc, char* argv[])
{
	int N = 16;
	if (argc > 1) {
		istringstream iss(argv[1]);
		if (iss >> N) {};
	}
	clock_t t0, t1;
	t0 = clock();
	//tryDMRG(N);
	//testHAB(N);
	//newDMRG(N);
	oopDMRG(N);
	//memoryTests(N);
	t1 = clock();
	printf("TOTAL TIME: %f SECONDS\n",double(t1-t0)/CLOCKS_PER_SEC);
	return 0;
}

