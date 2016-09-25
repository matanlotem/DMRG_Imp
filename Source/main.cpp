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
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <ctime>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "BDHamiltonian.hpp"
#include "BDMatrix.hpp"
#include "BODMatrix.hpp"

using namespace Eigen;
using namespace std;

#define DEBUG = false
#ifdef DEBUG
#define DBG(x) (x)
#else
#define DBG(x)
#endif




BDMatrix oopLanczos(BDHamiltonian *matrix, BDMatrix baseState) {
	int m = min(500, matrix->dim());
	int convIter = 5;
	double a,b2,norm;
	double tol = 0.0000001;

	BDMatrix prevState(baseState.blockNum()),
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
	tmpState = matrix->apply(currState); // H|u_n>
	a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>
	KMatrix(0,0) = a;

	prevState = currState;
	//currState = tmpState - a*currState;
	for (int b=0; b<currState.blockNum(); b++) currState[b] = tmpState[b] - a*currState[b];

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
		if (n%convIter == 0) {
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
	return outputState;
}

void oopDMRG(int N) {
	double Jxy=1, Jz=1, Hz=0;

	int n=2, b=2, D=256;
	BDMatrix HAPrev(b), SzAPrev(b),
			  HACurr(b), SzACurr(b),
			  DensityMatrix(b);
	BODMatrix SplusAPrev(b-1), SplusACurr(b-1);
	BDHamiltonian * HAB;
	SelfAdjointEigenSolver<MatrixXd> solver;
	vector<SelfAdjointEigenSolver<MatrixXd> > blockSolvers;
	vector<int> newBasisVectors;
	double baseEv = 0;

	// initialize operators
	SzAPrev[0] = (MatrixXd::Identity(1,1) * -0.5);
	SzAPrev[1] = (MatrixXd::Identity(1,1) *  0.5);

	for (int i=0; i<b; i++) {
		HAPrev[i]=Hz*SzAPrev[i];
		HAPrev.setBlockValue(i,SzAPrev[i](0,0));
		SzAPrev.setBlockValue(i,SzAPrev[i](0,0));
	}

	SplusAPrev.resize(&HAPrev);
	SplusAPrev[0] = MatrixXd::Identity(1,1);

	// grow inifinte-system DMRG
	while (n<N) {
		DBG(printf("Going to %d sites\n", n+2));
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

				//DBG(printf("\tadding spin down block\n"));
				SzACurr[blockInd] = MatrixXd::Identity(dim1,dim1) * -0.5;
				HACurr[blockInd] = HAPrev[i] + Jz * SzAPrev[i] * SzACurr[blockInd];
				for (int j=0; j<dim1; j++) HACurr[blockInd](j,j) += -0.5*Hz;
			}
			// if block with SzTot-1 exists
			else {
				HACurr.setBlockValue(blockInd,HAPrev.getBlockValue(i)-0.5);
				dim1 = HAPrev[i-1].rows();
				dim2 = HAPrev[i].rows();

				HACurr[blockInd] = MatrixXd(dim1+dim2,dim1+dim2);
				SzACurr[blockInd] = MatrixXd(dim1+dim2,dim1+dim2);
				SzACurr[blockInd].setZero();

				// add spin up block
				//DBG(printf("\tadding spin up block\n"));
				for (int j=0; j<dim1; j++) SzACurr[i](j,j) = 0.5;
				HACurr[blockInd].topLeftCorner(dim1,dim1) = HAPrev[i-1] +
							Jz * SzAPrev[i-1] * SzACurr[blockInd].topLeftCorner(dim1,dim1);
				for (int j=0; j<dim1; j++) HACurr[blockInd](j,j) += 0.5*Hz;

				// add spin down block
				//DBG(printf("\tadding spin down block\n"));
				for (int j=dim1; j<dim1+dim2; j++) SzACurr[blockInd](j,j) = -0.5;
				HACurr[blockInd].bottomRightCorner(dim2,dim2) = HAPrev[i] +
							Jz * SzAPrev[i] * SzACurr[blockInd].bottomRightCorner(dim2,dim2);
				for (int j=dim1; j<dim1+dim2; j++) HACurr[blockInd](j,j) += -0.5*Hz;

				// add old block movers Jxy/2 * (S+- + S-+)
				//DBG(printf("\tadding Jxy*(S+- + S-+) block\n"));
				HACurr[blockInd].topRightCorner(dim1,dim2) = Jxy/2 * SplusAPrev[i-1];
				HACurr[blockInd].bottomLeftCorner(dim2,dim1) = Jxy/2 * SplusAPrev[i-1].transpose();
			}
			blockInd++;

			// if last block or no block with SzTot+1 exists
			if ((i==HAPrev.blockNum() - 1) || (HAPrev.getBlockValue(i) + 1 != HAPrev.getBlockValue(i+1))) {
				HACurr.setBlockValue(blockInd,HAPrev.getBlockValue(i)+0.5);
				dim1 = HAPrev[i].rows();

				//DBG(printf("\tadding spin up block\n"));
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


		if (HACurr.rows() <= D) { // if operators are small enough, do not truncate
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
			BDMatrix baseStateMatrix(HAB->blockNum());
			int I, vecInd=0, blockInd=0;
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
			//baseStateMatrix.printFullStats();
			//HACurr.printFullStats();

			// find AB base state with Lanczos
			DBG(printf("finding AB base state with Lanczos\n"));
			BDMatrix ABBaseState = oopLanczos(HAB, baseStateMatrix);
			baseEv = ABBaseState.dot(HAB->apply(ABBaseState));
			DBG(printf("base Ev: %f\n", baseEv));

			// create density matrix
			DBG(printf("creating A density matrix\n"));
			DensityMatrix.resize(HAB->blockNum());
			for (int i=0; i<DensityMatrix.blockNum(); i++) {
				DensityMatrix[i] = ABBaseState[i]*ABBaseState[i].transpose();
				DensityMatrix.setBlockValue(i, ABBaseState.getBlockValue(i));
			}

			// diagonalize density matrix
			DBG(printf("diagonalizing A density matrix\n"));
			blockSolvers.resize(DensityMatrix.blockNum());
			for (int i=0; i<DensityMatrix.blockNum(); i++)
				blockSolvers[i].compute(DensityMatrix[i]);

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
				newBasisVectors[currBlock]++;
				vectorNum++;
			}


			// count truncated basis blocks;
			b = 0;
			for (int i=0; i<(int) newBasisVectors.size(); i++)
				if (newBasisVectors[i] > 0) b++;

			// transform operators to new basis
			DBG(printf("transforming matrices to new basis\n"));
			//DensityMatrix.printFullStats();
			//HACurr.printFullStats();

			SzAPrev.resize(b);
			HAPrev.resize(b);
			SplusAPrev.resize(b-1);

			blockInd = 0;
			int currInd;
			double blockValue;
			for (int i=0; i < (int) newBasisVectors.size(); i++) {
				if (newBasisVectors[i] > 0) {
					blockValue = DensityMatrix.getBlockValue(i);
					currInd = HACurr.getIndexByValue(blockValue);

					SzAPrev[blockInd] = blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]).transpose() *
							            SzACurr[currInd] * blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]);
					SzAPrev.setBlockValue(blockInd, blockValue);

					HAPrev[blockInd] = blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]).transpose() *
									   HACurr[currInd] * blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]);
					HAPrev.setBlockValue(blockInd, blockValue);

					if (blockInd>0)
						SplusAPrev[blockInd-1] = blockSolvers[i-1].eigenvectors().rightCols(newBasisVectors[i-1]).transpose() *
											     SplusACurr[currInd-1] * blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]);

					blockInd++;
				}
			}

			delete HAB;
		}

		n += 2;
	}

	printf("Base state eigenvalue for %d site Heisenberg model: %.20f\n", n, baseEv);
	printf("Bethe Ansatz eigenvalue in thermodynamical limit: %.20f\n", n*(0.25 - std::log(2.0)));
}

void FullDMRG(int N) {
	double Jxy=1, Jz=1, Hz=0;
	int D=128, maxSweeps=10;

	/* ******************************************************************************** */
	/* declare variables */
	/* ******************************************************************************** */

	// control variables
	int n = 2, b = 2, sweeps = 0, counter = 0;
	int currOp = 1, AOp = 0, BOp = N-1;
	bool infinite = true, done = false, growingA = true;
	double tolerance=0.0000000001;
	double baseEv;
	vector<double> baseEvFinite;
	double blockValue;

	// operators vectors
	vector<BDMatrix> H, Sz;
	vector<BODMatrix> Splus;

	// big hamiltonian
	BDHamiltonian * HAB;

	// base state
	VectorXd baseState;
	BDMatrix baseStateMatrix(1);
	BDMatrix ABBaseState(1);

	// desnity matrix
	BDMatrix DensityMatrix(1);
	vector<SelfAdjointEigenSolver<MatrixXd> > blockSolvers;
	vector<int> newBasisVectors;

	// basis transformation
	BDMatrix tmpH(1), tmpSz(1);
	BODMatrix tmpSplus(1);
	vector<BDMatrix> TransOp;


	/* ******************************************************************************** */
	/* initialize operators */
	/* ******************************************************************************** */

	for (int i=0; i<N; i++) {
		H.push_back(BDMatrix(2));
		Sz.push_back(BDMatrix(2));
		Splus.push_back(BODMatrix(1));
		TransOp.push_back(BDMatrix(2));
	}

	Sz[0][0] = MatrixXd::Identity(1,1) * -0.5;
	Sz[0][1] = MatrixXd::Identity(1,1) *  0.5;

	for (int i=0; i<b; i++) {
		H[0][i]=Hz*Sz[0][i];
		H[0].setBlockValue(i,Sz[0][i](0,0));
		Sz[0].setBlockValue(i,Sz[0][i](0,0));
	}

	Splus[0].resize(&H[0]);
	Splus[0][0] = MatrixXd::Identity(1,1);

	H[N-1] = H[0];
	Sz[N-1] = Sz[0];
	Splus[N-1] = Splus[0];

	tmpH = H[0];
	tmpSz = Sz[0];
	tmpSplus = Splus[0];

	/* ******************************************************************************** */
	/* start DMRG */
	/* ******************************************************************************** */
	while (!done) {
		counter++;
		/* ******************************************************************************** */
		/* grow operators (grow O' from O) O' index: currOp, O: tmpOp */
		/* ******************************************************************************** */
		// count new block number
		b = tmpH.blockNum() + 1;
		for (int i=1; i<tmpH.blockNum(); i++)
			if (tmpH.getBlockValue(i) - 1 != tmpH.getBlockValue(i-1)) b++;

		DBG(printf("Creating new blocks\n"));
		H[currOp].resize(b);
		Sz[currOp].resize(b);

		int dim1, dim2, blockInd = 0;
		for (int i=0; i<tmpH.blockNum(); i++) {
			// first block if no block with SzTot-1 exists
			if ((i==0) || (tmpH.getBlockValue(i) - 1 != tmpH.getBlockValue(i-1))) {
				H[currOp].setBlockValue(blockInd,tmpH.getBlockValue(i)-0.5);
				dim1 = tmpH[i].rows();

				Sz[currOp][blockInd] = MatrixXd::Identity(dim1,dim1) * -0.5;
				H[currOp][blockInd] = tmpH[i] + Jz * tmpSz[i] * Sz[currOp][blockInd];
				for (int j=0; j<dim1; j++) H[currOp][blockInd](j,j) += -0.5*Hz;
			}
			// if block with SzTot-1 exists
			else {
				H[currOp].setBlockValue(blockInd,tmpH.getBlockValue(i)-0.5);
				dim1 = tmpH[i-1].rows();
				dim2 = tmpH[i].rows();

				H[currOp][blockInd] = MatrixXd(dim1+dim2,dim1+dim2);
				Sz[currOp][blockInd] = MatrixXd(dim1+dim2,dim1+dim2);
				Sz[currOp][blockInd].setZero();

				// add spin up block
				for (int j=0; j<dim1; j++) Sz[currOp][i](j,j) = 0.5;
				H[currOp][blockInd].topLeftCorner(dim1,dim1) = tmpH[i-1] +
							Jz * tmpSz[i-1] * Sz[currOp][blockInd].topLeftCorner(dim1,dim1);
				for (int j=0; j<dim1; j++) H[currOp][blockInd](j,j) += 0.5*Hz;

				// add spin down block
				for (int j=dim1; j<dim1+dim2; j++) Sz[currOp][blockInd](j,j) = -0.5;
				H[currOp][blockInd].bottomRightCorner(dim2,dim2) = tmpH[i] +
							Jz * tmpSz[i] * Sz[currOp][blockInd].bottomRightCorner(dim2,dim2);
				for (int j=dim1; j<dim1+dim2; j++) H[currOp][blockInd](j,j) += -0.5*Hz;

				// add old block movers Jxy/2 * (S+- + S-+)
				H[currOp][blockInd].topRightCorner(dim1,dim2) = Jxy/2 * tmpSplus[i-1];
				H[currOp][blockInd].bottomLeftCorner(dim2,dim1) = Jxy/2 * tmpSplus[i-1].transpose();
			}
			blockInd++;

			// if last block or no block with SzTot+1 exists
			if ((i==tmpH.blockNum() - 1) || (tmpH.getBlockValue(i) + 1 != tmpH.getBlockValue(i+1))) {
				H[currOp].setBlockValue(blockInd,tmpH.getBlockValue(i)+0.5);
				dim1 = tmpH[i].rows();

				Sz[currOp][blockInd] = MatrixXd::Identity(dim1,dim1) * 0.5;
				H[currOp][blockInd] = tmpH[i] + Jz * tmpSz[i] * Sz[currOp][blockInd];
				for (int j=0; j<dim1; j++) H[currOp][blockInd](j,j) += 0.5*Hz;

				blockInd++;
			}
		}

		//create new block movers (S+)
		DBG(printf("creating new S+ blocks\n"));
		Splus[currOp].resize(&H[currOp]);
		for (int i=0; i<Splus[currOp].blockNum(); i++) {
			if (H[currOp].getBlockValue(i)+1 == H[currOp].getBlockValue(i+1)) {
				dim1 = tmpH[tmpH.getIndexByValue(H[currOp].getBlockValue(i)+0.5)].rows();
				dim2 = H[currOp][i].rows();

				if ((dim1==dim2) && (dim1 == H[currOp][i+1].cols()))
					Splus[currOp][i] = MatrixXd::Identity(dim1,dim1);
				else {
					Splus[currOp][i].setZero();
					for (int j=0; j<dim1;j++) Splus[currOp][i](dim2-dim1+j,j) = 1;
				}
			}
			else // if no block with SzTot+1 exists
				Splus[currOp][i].setZero();

		}

		/* ******************************************************************************** */
		/* controller */
		/* ******************************************************************************** */
		// for inifinte-system DMRG mirror A operators to get B operators
		if (infinite) {
			AOp = currOp;
			BOp = N - 1 - AOp;

			H[BOp] = H[AOp];
			Sz[BOp] = Sz[AOp];
			Splus[BOp] = Splus[AOp];
		}


		if (H[currOp].rows() <= D) {
			tmpH = H[currOp];
			tmpSz = Sz[currOp];
			tmpSplus = Splus[currOp];
		}

		else { //truncate
			/* ******************************************************************************** */
			/* create AB Hamiltonian (using A & B opertators) */
			/* ******************************************************************************** */
			int SzTot = 0;
			DBG(printf("creating AB Hamiltonian\n"));
			HAB = new BDHamiltonian(&H[AOp], &Sz[AOp], &Splus[AOp], &H[BOp], &Sz[BOp], &Splus[BOp], Jxy, Jz, Hz, SzTot);

			/* ******************************************************************************** */
			/* guess AB base state */
			/* ******************************************************************************** */
			DBG(printf("creating AB initial base state\n"));
			if (infinite || !infinite) {
				baseState.resize(HAB->dim());
				baseState.setRandom();
				baseState = baseState / baseState.norm();
			}

			/*if (!infinite) {
				printf("H AOp BOp\n");
				H[AOp].printFullStats();
				H[BOp].printFullStats();
				printf("New base state\n");
				baseStateMatrix.printFullStats();
				printf("Old base state\n");
				ABBaseState.printFullStats();
				printf("TransOp -1 0 +1\n");
				TransOp[currOp-1].printFullStats();
				TransOp[currOp].printFullStats();
				TransOp[currOp+1].printFullStats();
				printf("currOp=%d, AOp=%d, BOp=%d\n",currOp, AOp, BOp);

			}*/

			baseStateMatrix.resize(HAB->blockNum());
			int I, vecInd=0, blockInd=0, LInd, RInd;
			for (int i=0; i<H[AOp].blockNum(); i++) {
				blockValue = H[AOp].getBlockValue(i);
				I = H[BOp].getIndexByValue(SzTot - blockValue);
				if (I!=-1) {
					baseStateMatrix[blockInd].resize(H[AOp][i].rows(), H[BOp][I].rows());
					baseStateMatrix.setBlockValue(blockInd, H[AOp].getBlockValue(i));
					if (infinite || TransOp[currOp+1].rows() <= D || TransOp[currOp-1].rows() <= D) {
						for (int j=0; j<H[AOp][i].rows(); j++)
							for (int k=0; k<H[BOp][I].rows(); k++)
								baseStateMatrix[blockInd](j,k) = baseState(vecInd++);
					}
					else {
						//printf("i=%d, I=%d, blockInd=%d, blockValue=%f\n",i,I,blockInd,blockValue);

						if (growingA) {
							//printf("block dim: %dx%d\n", (int) baseStateMatrix[blockInd].rows(), (int) baseStateMatrix[blockInd].cols());
							//baseStateMatrix[blockInd].setZero();
							LInd = TransOp[currOp-1].getIndexByValue(blockValue - 0.5);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue);
							//printf("LInd=%d, RInd=%d\n",LInd, RInd);
							if (LInd != -1 && RInd != -1)
								baseStateMatrix[blockInd].topRows(TransOp[currOp-1][LInd].cols()) =
										TransOp[currOp-1][LInd].transpose() *
										ABBaseState[ABBaseState.getIndexByValue(blockValue - 0.5)].leftCols(TransOp[currOp+1][RInd].cols()) *
										TransOp[currOp+1][RInd].transpose();

							LInd = TransOp[currOp-1].getIndexByValue(blockValue + 0.5);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue);
							//printf("LInd=%d, RInd=%d\n",LInd, RInd);
							if (LInd != -1 && RInd != -1)
								baseStateMatrix[blockInd].bottomRows(TransOp[currOp-1][LInd].cols()) =
										TransOp[currOp-1][LInd].transpose() *
										ABBaseState[ABBaseState.getIndexByValue(blockValue + 0.5)].rightCols(TransOp[currOp+1][RInd].cols()) *
										TransOp[currOp+1][RInd].transpose();
						}
						else {
							//printf("block %d (%.1f) dim: %dx%d\n", blockInd, blockValue, (int) baseStateMatrix[blockInd].rows(), (int) baseStateMatrix[blockInd].cols());
							baseStateMatrix[blockInd].setZero();
							LInd = TransOp[currOp-1].getIndexByValue(blockValue);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue + 0.5);
							//printf("LInd=%d, RInd=%d\n",LInd, RInd);
							if (LInd != -1 && RInd != -1)
								baseStateMatrix[blockInd].rightCols(TransOp[currOp+1][RInd].cols()) =
										TransOp[currOp-1][LInd] *
										ABBaseState[ABBaseState.getIndexByValue(blockValue - 0.5)].bottomRows(TransOp[currOp-1][LInd].cols()) *
										TransOp[currOp+1][RInd];

							LInd = TransOp[currOp-1].getIndexByValue(blockValue);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue - 0.5);
							//printf("LInd=%d, RInd=%d\n",LInd, RInd);
							if (LInd != -1 && RInd != -1)
								baseStateMatrix[blockInd].leftCols(TransOp[currOp+1][RInd].cols()) =
										TransOp[currOp-1][LInd] *
										ABBaseState[ABBaseState.getIndexByValue(blockValue + 0.5)].topRows(TransOp[currOp-1][LInd].cols()) *
										TransOp[currOp+1][RInd];
						}

					}

					blockInd++;
				}
			}
			/*if (!(infinite || TransOp[currOp+1].rows() <= D || TransOp[currOp-1].rows() <= D) && !growingA) {
				baseStateMatrix.printFullStats();
				printf("%d\n",counter);
				if (counter>24)
					abort();
			}*/

			/*printf("old base state norm = %f\n", ABBaseState.norm());
			for (int i=0; i< ABBaseState.blockNum(); i++)
				printf("%f  ", ABBaseState[i].norm());
			printf("\n");
			printf("new base state norm = %f\n", baseStateMatrix.norm());
			for (int i=0; i< baseStateMatrix.blockNum(); i++)
				printf("%f  ", baseStateMatrix[i].norm());
			printf("\n");*/
			//baseStateMatrix.printFullStats();

			/* ******************************************************************************** */
			/* find AB base state with Lanczos */
			/* ******************************************************************************** */
			DBG(printf("finding AB base state with Lanczos\n"));
			ABBaseState = oopLanczos(HAB, baseStateMatrix);
			baseEv = ABBaseState.dot(HAB->apply(ABBaseState));
			DBG(printf("base Ev: %.20f\n", baseEv));

			/* ******************************************************************************** */
			/* density matrix - calculate, diagonalize, choose new basis */
			/* ******************************************************************************** */
			//create density matrix
			DBG(printf("creating density matrix\n"));
			DensityMatrix.resize(HAB->blockNum());
			if (growingA) {
				for (int i=0; i<DensityMatrix.blockNum(); i++) {
					DensityMatrix[i] = ABBaseState[i]*ABBaseState[i].transpose();
					DensityMatrix.setBlockValue(i, ABBaseState.getBlockValue(i));
				}
			}
			else {
				for (int i=0; i<DensityMatrix.blockNum(); i++) {
					// reverse order so Sz is sorted in acceding order
					DensityMatrix[DensityMatrix.blockNum()-i-1] = ABBaseState[i].transpose()*ABBaseState[i];
					DensityMatrix.setBlockValue(DensityMatrix.blockNum()-i-1, SzTot - ABBaseState.getBlockValue(i));
				}
			}

			// diagonalize density matrix
			blockSolvers.resize(DensityMatrix.blockNum());
			DBG(printf("diagonalizing density matrix\n"));
			for (int i=0; i<DensityMatrix.blockNum(); i++)
				blockSolvers[i].compute(DensityMatrix[i]);

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
				newBasisVectors[currBlock]++;
				vectorNum++;
			}

			// count truncated basis blocks;
			b = 0;
			for (int i=0; i<(int) newBasisVectors.size(); i++)
				if (newBasisVectors[i] > 0) b++;

			// create transformation opertator matrix
			TransOp[currOp].resize(b);
			blockInd = 0;
			for (int i=0; i<(int) newBasisVectors.size(); i++) {
				if (newBasisVectors[i] > 0) {
					TransOp[currOp][blockInd] = blockSolvers[i].eigenvectors().rightCols(newBasisVectors[i]);
					TransOp[currOp].setBlockValue(blockInd, DensityMatrix.getBlockValue(i));
					blockInd++;
				}
			}
			if (infinite) TransOp[BOp] = TransOp[AOp];


			/* ******************************************************************************** */
			/* transform operators to new basis */
			/* ******************************************************************************** */
			DBG(printf("transforming matrices to new basis\n"));

			tmpSz.resize(b);
			tmpH.resize(b);
			tmpSplus.resize(b-1);

			int currInd;

			for (int i=0; i<TransOp[currOp].blockNum(); i++) {
				blockValue = TransOp[currOp].getBlockValue(i);
				currInd = H[currOp].getIndexByValue(blockValue);

				tmpSz[i] = TransOp[currOp][i].transpose() * Sz[currOp][currInd] * TransOp[currOp][i];
				tmpH[i] = TransOp[currOp][i].transpose() * H[currOp][currInd] * TransOp[currOp][i];
				if (i>0) tmpSplus[i-1] = TransOp[currOp][i-1].transpose() * Splus[currOp][currInd - 1] * TransOp[currOp][i];
				tmpSz.setBlockValue(i, blockValue);
				tmpH.setBlockValue(i, blockValue);

			}

			delete HAB;
		}

		/* ******************************************************************************** */
		/* controller */
		/* ******************************************************************************** */
		DBG(printf("previous operator indices: AOp=%d, BOp=%d, currOp=%d\n",AOp,BOp,currOp));
		//printf("   H[AOp]:  "); H[AOp].printStats();
		//printf("   H[BOp]:  "); H[BOp].printStats();

		if (infinite) {
			n += 2;
			if (AOp+1 == BOp) {
				infinite = false; // infinite-system DMRG is done

				//create dummy transformation operators
				for (int i=0; i<N/2; i++) {
					if (H[i].rows() <= D) {
						TransOp[i] = H[i];
						for(int j=0; j<TransOp[i].blockNum(); j++)
							TransOp[i][j].setIdentity();
						TransOp[N-i-1] = TransOp[i];
					}
				}

				for (int i=0; i<N; i++) H[i].printStats();
				printf("\n");
				for (int i=0; i<N; i++) {
					printf("%d  ",i);
					H[i].printFullStats();
				}
				printf("\n");
				//abort();
			}
			else {
				currOp++;

				DBG(printf("Going to %d sites\n", n+2));
			}
		}

		if (!infinite) {
			// sweeps control
			if ((AOp + BOp + 1) == N) {
				// done if maxSweeps or baseEv converges
				done = ((sweeps >= maxSweeps) ||
						(sweeps > 1 && abs(baseEv - baseEvFinite[baseEvFinite.size()-1])<tolerance));

				baseEvFinite.push_back(baseEv);
				sweeps++;
			}


			// move right (grow A)
			// //***goes one extra step
			if ((growingA && (N-1-BOp >= log(D)/log(2))) ||
			    (!growingA && (AOp < log(D)/log(2)))) {

				if (!growingA) {
					tmpH = H[AOp];
					tmpSz = Sz[AOp];
					tmpSplus = Splus[AOp];
					growingA = true;
				}

				AOp++;
				BOp++;
				currOp = AOp;
			}

			// move left (grow B)
			else {
				if (growingA) {
					tmpH = H[BOp];
					tmpSz = Sz[BOp];
					tmpSplus = Splus[BOp];
					growingA = false;
				}
				BOp--;
				AOp--;
				currOp = BOp;
			}
		}
		DBG(printf("next operator indices: AOp=%d, BOp=%d, currOp=%d\n",AOp,BOp,currOp));
		//printf("   H[AOp]:  "); H[AOp].printStats();
		//printf("   H[BOp]:  "); H[BOp].printStats();

	}
	printf("Base state eigenvalue for %d site Heisenberg model IDMRG: %.20f\n", n, baseEvFinite[0]);
	printf("Base state eigenvalue for %d site Heisenberg model FDMRG: %.20f\n", n, baseEvFinite[baseEvFinite.size()-1]);
	printf("Bethe Ansatz eigenvalue in thermodynamical limit: %.20f\n", n*(0.25 - std::log(2.0)));
	printf("Total sweeps: %d\n",sweeps);
	printf("Total iterations: %d\n",counter);

	ofstream file;
	file.open("../Output/tmp.xls");
	for (int i=0; i < (int) baseEvFinite.size(); i++)
		file << std::fixed << std::setprecision(20) << baseEvFinite[i] << endl;
	file.close();

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
	FullDMRG(N);
	t1 = clock();
	printf("TOTAL TIME: %f SECONDS\n",double(t1-t0)/CLOCKS_PER_SEC);
	return 0;
}

