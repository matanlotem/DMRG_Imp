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

/*#define DEBUG = false
#ifdef DEBUG
#define DBG(x) (x)
#else*/
#define DBG(x)
//#endif




BDMatrix oopLanczos(BDHamiltonian *matrix, BDMatrix baseState, bool keepVectors, int convIter, double tol) {
	int m = min(500, matrix->dim());
	double a,b2,norm;

	BDMatrix prevState(baseState.blockNum()),
			  currState(baseState.blockNum()),
			  tmpState(baseState.blockNum()),
			  outputState(baseState.blockNum());

	vector<BDMatrix> allStates;

	MatrixXd KMatrix(m,m);
	SelfAdjointEigenSolver<MatrixXd> solver;
	KMatrix.setZero();

	//first iteration
	currState = baseState;
	if (keepVectors) allStates.push_back(currState);

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
		if (keepVectors) allStates.push_back(currState);
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
			solver.compute(KMatrix.block(0,0,n,n),false);
			currEv= solver.eigenvalues()[0];
			converged = abs(currEv - prevEv) < tol;
			prevEv = currEv;
		}
	}

	//printf("%d iterations\n",n);
	if (n<m) {
		MatrixXd tmpKMatrix = KMatrix.block(0,0,n,n);
		KMatrix = tmpKMatrix;
		m = n;
	}

	//diagonalize
	solver.compute(KMatrix);

	// calculate eigenvector
	VectorXd minEigenVector = solver.eigenvectors().col(0);


	if (keepVectors) {
		for (int b=0; b<outputState.blockNum(); b++) outputState[b] = 0 * baseState[b];

		for (int n=0; n<m; n++)
			for (int b=0; b<outputState.blockNum(); b++)
				outputState[b] += allStates[n][b] / allStates[n].norm() * minEigenVector(n);
	}

	else {
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
	}

	for (int i=0; i<baseState.blockNum(); i++) outputState.setBlockValue(i, baseState.getBlockValue(i));
	return outputState;
}

BDMatrix oopLanczos(BDHamiltonian *matrix, BDMatrix baseState) {
	return oopLanczos(matrix, baseState, true, 5, 0.00000001);
}


void FullDMRG(int N) {
	double Jxy=1, Jz=1, Hz=0;
	int D=128, maxSweeps=20;
	clock_t l0, l1, t0, t1, s0, s1;
	double li_tot = 0, lf_tot = 0, d_tot = 0, t_tot = 0, i_tot, f_tot;

	/* ******************************************************************************** */
	/* declare variables */
	/* ******************************************************************************** */

	// control variables
	int n = 2, b = 2, sweeps = 0, counter = 0;
	int currOp = 1, AOp = 0, BOp = N-1;
	bool infinite = true, done = false, growingA = true;
	double tolerance=0.00000001;
	bool lanczosKeepVectors = true;
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
	t0 = clock();
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
		// for Infinite-System DMRG mirror A operators to get B operators
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
			/* generate AB base state */
			/* ******************************************************************************** */
			DBG(printf("creating AB initial base state\n"));
			// if Infinite-System DMRG or Finite-System DMRG turning point => random base state
			if (infinite || TransOp[currOp+1].rows() <= D || TransOp[currOp-1].rows() <= D) {
				baseState.resize(HAB->dim());
				baseState.setRandom();
				baseState = baseState / baseState.norm();
			}

			baseStateMatrix.resize(HAB->blockNum());
			int I, vecInd=0, blockInd=0, LInd, RInd;
			for (int i=0; i<H[AOp].blockNum(); i++) {
				blockValue = H[AOp].getBlockValue(i);
				I = H[BOp].getIndexByValue(SzTot - blockValue);
				if (I!=-1) {
					baseStateMatrix[blockInd].resize(H[AOp][i].rows(), H[BOp][I].rows());
					baseStateMatrix.setBlockValue(blockInd, H[AOp].getBlockValue(i));
					// random base state
					if (infinite || TransOp[currOp+1].rows() <= D || TransOp[currOp-1].rows() <= D) {
						for (int j=0; j<H[AOp][i].rows(); j++)
							for (int k=0; k<H[BOp][I].rows(); k++)
								baseStateMatrix[blockInd](j,k) = baseState(vecInd++);
					}
					// transform prevoius base state
					else {
						if (growingA) {
							baseStateMatrix[blockInd].setZero();
							LInd = TransOp[currOp-1].getIndexByValue(blockValue - 0.5);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue);
							if (LInd != -1 && RInd != -1)
								baseStateMatrix[blockInd].topRows(TransOp[currOp-1][LInd].cols()) =
										TransOp[currOp-1][LInd].transpose() *
										ABBaseState[ABBaseState.getIndexByValue(blockValue - 0.5)].leftCols(TransOp[currOp+1][RInd].cols()) *
										TransOp[currOp+1][RInd].transpose();

							LInd = TransOp[currOp-1].getIndexByValue(blockValue + 0.5);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue);
							if (LInd != -1 && RInd != -1)
								baseStateMatrix[blockInd].bottomRows(TransOp[currOp-1][LInd].cols()) =
										TransOp[currOp-1][LInd].transpose() *
										ABBaseState[ABBaseState.getIndexByValue(blockValue + 0.5)].rightCols(TransOp[currOp+1][RInd].cols()) *
										TransOp[currOp+1][RInd].transpose();
						}
						else {
							baseStateMatrix[blockInd].setZero();
							LInd = TransOp[currOp-1].getIndexByValue(blockValue);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue + 0.5);
							if (LInd != -1 && RInd != -1)
								baseStateMatrix[blockInd].rightCols(TransOp[currOp+1][RInd].cols()) =
										TransOp[currOp-1][LInd] *
										ABBaseState[ABBaseState.getIndexByValue(blockValue - 0.5)].bottomRows(TransOp[currOp-1][LInd].cols()) *
										TransOp[currOp+1][RInd];

							LInd = TransOp[currOp-1].getIndexByValue(blockValue);
							RInd = TransOp[currOp+1].getIndexByValue(SzTot - blockValue - 0.5);
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

			/* ******************************************************************************** */
			/* find AB base state with Lanczos */
			/* ******************************************************************************** */
			DBG(printf("finding AB base state with Lanczos\n"));
			l0 = clock();
			if (infinite) {
				ABBaseState = oopLanczos(HAB, baseStateMatrix, lanczosKeepVectors, 5, tolerance);
				l1 = clock();
				li_tot += double(l1-l0)/CLOCKS_PER_SEC;
			}
			else {
				ABBaseState = oopLanczos(HAB, baseStateMatrix, lanczosKeepVectors, 1, tolerance);
				l1 = clock();
				lf_tot += double(l1-l0)/CLOCKS_PER_SEC;
			}
			l0 = clock();
			baseEv = ABBaseState.dot(HAB->apply(ABBaseState));
			l1 = clock();
			if (!infinite) t_tot += double(l1-l0);
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
			l0 = clock();
			// diagonalize density matrix
			blockSolvers.resize(DensityMatrix.blockNum());
			DBG(printf("diagonalizing density matrix\n"));
			for (int i=0; i<DensityMatrix.blockNum(); i++)
				blockSolvers[i].compute(DensityMatrix[i]);
			l1 = clock();
			d_tot += double(l1-l0);

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

		if (infinite) {
			n += 2;
			if (AOp+1 == BOp) {
				infinite = false; // infinite-system DMRG is done
				t1 = clock();
				i_tot = double(t1-t0)/CLOCKS_PER_SEC;
				t0 = t1;
				s0 = t0;
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
				//done = ((sweeps >= maxSweeps) ||
				//		(sweeps > 1 && abs(baseEv - baseEvFinite[baseEvFinite.size()-1])<tolerance));
				done = sweeps > 10;

				baseEvFinite.push_back(baseEv);
				s1 = clock();
				printf("%f Seconds\n",double(s1-s0)/CLOCKS_PER_SEC);
				sweeps++;
				s0 = s1;
			}


			// move right (grow A)
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
	t1 = clock();
	f_tot = double(t1-t0)/CLOCKS_PER_SEC;

	printf("Base state eigenvalue for %d site Heisenberg model IDMRG: %.20f\n", n, baseEvFinite[0]);
	printf("Base state eigenvalue for %d site Heisenberg model FDMRG: %.20f\n", n, baseEvFinite[baseEvFinite.size()-1]);
	printf("Bethe Ansatz eigenvalue in thermodynamical limit: %.20f\n", n*(0.25 - std::log(2.0)));
	printf("Total sweeps: %d\n",sweeps);
	printf("Total iterations: %d\n",counter);

	printf("\n");
	printf("infinite: %f SECONDS\n",i_tot);
	printf("\tlanczos: %f SECONDS\n",li_tot);
	printf("finite: %f SECONDS\n",f_tot);
	printf("\tlanczos: %f SECONDS\n",lf_tot);
	printf("\tdensity: %f SECONDS\n",d_tot / CLOCKS_PER_SEC);
	printf("\ttransofrm: %f SECONDS\n",t_tot / CLOCKS_PER_SEC);
	printf("\n");


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

