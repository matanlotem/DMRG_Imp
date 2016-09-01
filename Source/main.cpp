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
//#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "utils.hpp"
#include "ABHamiltonian.hpp"
#include "ABHamiltonianD3.hpp"
#include "ABHamiltonianFull.hpp"

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
	MatrixXd SzTmpBlock;
	double *SzPrevTot, *SzCurrTot;

	int n=1, b = 2, dim, dim1, dim2, dim3;
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


		DBG(printf("Creating new central blocks\n"));
		for (int i=1; i<b-1; i++) {
			DBG(printf("\tblock %d\n",i));
			dim1 = HAPrevBlocks[i-1].rows();
			dim2 = HAPrevBlocks[i].rows();
			HACurrBlocks[i] = MatrixXd(dim1+dim2,dim1+dim2);
			HACurrBlocks[i].setZero();
			SzACurrBlocks[i] = MatrixXd(dim1+dim2,dim1+dim2);
			SzACurrBlocks[i].setZero();

			DBG(printf("\tadding spin down block\n"));
			for (int j=0; j<dim1; j++) SzACurrBlocks[i](j,j) = 0.5;
			HACurrBlocks[i].block(0,0,dim1,dim1) = HAPrevBlocks[i-1] +
					Jz * SzAPrevBlocks[i-1] * SzACurrBlocks[i].block(0,0,dim1,dim1);
			for (int j=0; j<dim1; j++) HACurrBlocks[i](j,j) += 0.5*Hz;

			//SzTmpBlock = Jz * SzAPrevBlocks[i-1] * SzACurrBlocks[i].block(0,0,dim1,dim1);
			/*for (int j=0; j<dim1; j++) {
				//for (int k=0; k<dim1; k++)
				//	HACurrBlocks[i](j,k) = HAPrevBlocks[i-1](j,k) + SzTmpBlock(j,k);
				HACurrBlocks[i](j,j) += 0.5*Hz;
			}*/


			DBG(printf("\tadding spin up block\n"));
			for (int j=0; j<dim2; j++) SzACurrBlocks[i](dim1+j,dim1+j) = -0.5;
			HACurrBlocks[i].block(dim1,dim1,dim2,dim2) = HAPrevBlocks[i] +
						Jz * SzAPrevBlocks[i] * SzACurrBlocks[i].block(dim1,dim1,dim2,dim2);
			for (int j=0; j<dim2; j++) HACurrBlocks[i](dim1+j,dim1+j) += -0.5*Hz;

			/*SzTmpBlock = Jz * SzAPrevBlocks[i] * SzACurrBlocks[i].block(dim1,dim1,dim2,dim2);
			for (int j=0; j<dim2; j++) {
				for (int k=0; k<dim2; k++)
					HACurrBlocks[i](dim1+j,dim1+k) = HAPrevBlocks[i](j,k) + SzTmpBlock(j,k);
				HACurrBlocks[i](dim1+j,dim1+j) += -0.5*Hz;
			}*/


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

		DBG(printf("printing HA\n"));
		dim = 0;
		for (int i=0; i<b; i++) dim += HACurrBlocks[i].rows();
		HAMatrix.resize(dim,dim);
		HAMatrix.setZero();
		int ind = 0;
		for (int i=0; i<b; i++) {
			dim1 = HACurrBlocks[i].rows();
			for (int j=0; j<dim1; j++)
				for (int k=0; k<dim1; k++)
					HAMatrix(ind+j,ind+k) = HACurrBlocks[i](j,k);
			ind+=dim1;
		}
		cout << HAMatrix << endl << endl;
		solver.compute(HAMatrix,false);
		cout << solver.eigenvalues().transpose() << endl << endl;

		delete[] HAPrevBlocks;
		delete[] SzAPrevBlocks;
		delete[] SplusAPrevBlocks;
		delete SzPrevTot;
		HAPrevBlocks = HACurrBlocks;
		SzAPrevBlocks = SzACurrBlocks;
		SplusAPrevBlocks = SplusACurrBlocks;
		SzPrevTot = SzCurrTot;

		n++;
	}
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
	newDMRG(N);
	t1 = clock();
	printf("TOTAL TIME: %f SECONDS\n",double(t1-t0)/CLOCKS_PER_SEC);
	return 0;
}

