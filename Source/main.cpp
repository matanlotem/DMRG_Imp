/*
 * main.cpp
 *
 *  Created on: Aug 24, 2016
 *      Author: Matan
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define DEBUG = true
#ifdef DEBUG
#define DBG(x) (x)
#else
#define DBG(x)
#endif

MatrixXd tensorProdIdent(MatrixXd matrixA, int identDim) {
	MatrixXd outputMatrix(matrixA.rows()*identDim, matrixA.cols()*identDim);
	outputMatrix.setZero();
	for (int i=0; i<matrixA.rows(); i++)
		for (int j=0; j<matrixA.cols(); j++)
			for (int k=0; k<identDim; k++)
				outputMatrix(i*identDim + k,j*identDim + k) = matrixA(i,j);
	return outputMatrix;
}

MatrixXd tensorProdIdent(int identDim, MatrixXd matrixB) {
	MatrixXd outputMatrix(matrixB.rows()*identDim, matrixB.cols()*identDim);
	outputMatrix.setZero();
	for (int k=0; k<identDim; k++)
		for (int i=0; i<matrixB.rows(); i++)
			for (int j=0; j<matrixB.cols(); j++)
				outputMatrix(k*matrixB.rows() + i, k*matrixB.cols() + j) = matrixB(i,j);
	return outputMatrix;
}

MatrixXd copyResize(MatrixXd matrix, int newRows, int newCols) {
	MatrixXd newMatrix(newRows,newCols);
	int rows = min((int) matrix.rows(),newRows);
	int cols = min((int) matrix.cols(),newCols);
	newMatrix.setZero();
	for (int i=0; i < rows; i++)
		for (int j=0; j < cols; j++)
			newMatrix(i,j) = matrix(i,j);
	return newMatrix;
}

MatrixXd copyResize(MatrixXd matrix, int newDim) {
	return copyResize(matrix, newDim, newDim);
}

VectorXd naiveLanczos(MatrixXd matrix) {
	int dim = matrix.cols();
	int m = min(500, dim);
	double a,b2,norm;
	VectorXd baseState(dim), outputState(dim), prevState(dim), currState(dim), tmpState(dim);
	// generate base state
	baseState.setRandom();
	baseState = baseState / baseState.norm();

	MatrixXd KMatrix(m,m), tmpKMatrix;
	SelfAdjointEigenSolver<MatrixXd> solver, tmpSolver;
	KMatrix.setZero();

	//first iteration
	currState = baseState;
	norm = currState.norm();
	norm = norm*norm; // <u_n|u_n>
	tmpState = matrix*currState; // H|u_n>
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

		tmpState = matrix*currState; // H|u_n>
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
			tmpKMatrix = copyResize(KMatrix,n);
			tmpSolver.compute(tmpKMatrix,false);
			currEv= tmpSolver.eigenvalues()[0];
			converged = abs(currEv -prevEv) < 0.00000000000001;
			prevEv = currEv;
		}
	}

	printf("%d iterations\n",n);
	m = n;

	//diagonalize
	solver.compute(KMatrix);

	// calculate eigenvector
	VectorXd minEigenVector = solver.eigenvectors().col(0);

	currState = baseState;
	outputState = currState / currState.norm() * minEigenVector(0);

	norm = currState.norm();
	norm = norm*norm; // <u_n|u_n>
	tmpState = matrix*currState; // H|u_n>
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

		tmpState = matrix*currState; // H|u_n>
		a = tmpState.dot(currState) / norm; // <u_n|H|u_n>/<u_n|u_n>

		tmpState = tmpState - a*currState - b2*prevState;
		prevState = currState;
		currState = tmpState;

		n++;
	}

	return outputState;
}

void tryDMRG() {
	int N = 256;
	double Jxy=1, Jz=1, Hz=0;

	MatrixXd SzA1(2,2), SminA1(2,2), SplusA1(2,2),
			 SzAPrev, SminAPrev, SplusAPrev, HAPrev,
			 SzACurr, SminACurr, SplusACurr, HA, HAB,
			 densityMatrix, transNew2Old, transOld2New;
	VectorXd ABBaseState;
	SelfAdjointEigenSolver<MatrixXd> densitySolver;

	int D = 2, n = 2, chi = 8;

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
			HAB = tensorProdIdent(HA,D) + tensorProdIdent(D, HA.reverse())
					+ Jxy/2 * tensorProdIdent(SminACurr,D)*tensorProdIdent(D,SplusACurr.reverse())
					+ Jxy/2 * tensorProdIdent(SplusACurr,D)*tensorProdIdent(D,SminACurr.reverse())
					+ Jz * tensorProdIdent(SzACurr,D)*tensorProdIdent(D,SzACurr.reverse());

			// find AB base state using Lanczos
			DBG(printf("\tfinding AB base state with Lanczos\n"));
			ABBaseState = naiveLanczos(HAB);
			//DBG(printf("\tn=%d min eigenVector norm=%.15f, H*ev norm = %.15f\n",n, ABBaseState.norm(), (HAB*ABBaseState).norm()));
			DBG(printf("\tn=%d min eigenvalue: %.15f\n",n, (HAB*ABBaseState).norm()));

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
			//transNew2Old = copyResize(densitySolver.eigenvectors(),D,chi);
			transNew2Old.resize(D,chi);
			for (int i=0; i<D; i++)
				for (int j=0; j<chi; j++)
					transNew2Old(i,j) = densitySolver.eigenvectors()(i,D-chi+j);
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

		n += 2;
	}
	printf("Done: n=%d D=%d\n\n",n,D);
	/*cout << HA  << endl << endl;

	SelfAdjointEigenSolver<MatrixXd> solver(HA);
	VectorXd outputState = naiveLanczos(HA);
	cout << "Lanczos output state: " << outputState.transpose() << endl << endl;
	cout << "HA*(output state): " << (HA*outputState).transpose() << endl << endl;

	cout << "Exact output state: " << solver.eigenvectors().col(0).transpose() << endl << endl;
	cout << "HA*(output state): " << (HA*solver.eigenvectors().col(0)).transpose() << endl << endl;*/

	/*// find AB base state using Lanczos
	cout << HA  << endl << endl;
	//naiveLanczos(HA);
	//cout << HAB  << endl << endl;
	ABBaseState = naiveLanczos(HA);
	std::cout << ABBaseState.transpose() << endl <<endl;
	std::cout << (HA * ABBaseState).transpose() << endl <<endl;


	//cout << HA.reverse()  << endl << endl;*/
}

int main(int argc, char* argv[])
{
	tryDMRG();
	return 0;
}

