/*
 * BDMatrix.cpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include "BDMatrix.hpp"

using namespace Eigen;
using namespace std;

BDMatrix::BDMatrix(int b) : b(b) {
	//blocks = new MatrixXd[b];
	//blockValues = new double[b];
	blocks.resize(b);
	blockValues.resize(b);
	for (int i=0; i<b; i++) blocks[i] = MatrixXd(0,0);
}

BDMatrix::BDMatrix(BDMatrix * matrix, VectorXd * vector) {
	resize(blockNum());
	int ind=0, rows, cols;
	for (int i=0; i<b; i++){
		rows = (*matrix)[i].rows();
		cols = (*matrix)[i].cols();

		blocks[i] = MatrixXd(rows,cols);
		for (int j=0; j<rows; j++)
			for (int k=0; k<cols; k++)
				blocks[i](j,k) = (*vector)(j*rows+k+ind);
		ind+=rows*cols;
	}

	if (ind != matrix->rows() * matrix->cols()) {
		printf("ind=%d  matrix: %dx%d\n",ind, matrix->rows(), matrix->cols());
		throw "matrix-vector dimensions don't match!";
		abort();
	}

}

BDMatrix::~BDMatrix() {
	//printf("Destructor %d started\n",this);
	//delete[] blocks;
	//delete[] blockValues;
	//printf("Destructor %d done\n",this);
}

int BDMatrix::blockNum() {return b;}

int BDMatrix::rows() {
	int d = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].rows() > 0)
			d += blocks[i].rows();
	return d;
}

int BDMatrix::cols() {
	int d = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].cols() > 0)
			d += blocks[i].cols();
	return d;
}

MatrixXd& BDMatrix::operator[] (const int index) {
	return blocks[index];
}

MatrixXd BDMatrix::toMatrix() {
	MatrixXd matrix(rows(),cols());
	matrix.setZero();
	int indRows = 0, indCols = 0;
	for (int i=0; i<blockNum(); i++) {
		matrix.block(indRows,indCols,blocks[i].rows(),blocks[i].cols()) = blocks[i];
		indRows += blocks[i].rows();
		indCols += blocks[i].cols();
	}
	return matrix;
}

void BDMatrix::print() {
	cout << toMatrix() << endl;
}

void BDMatrix::printStats() {
	printf("dim: %dx%d, blocks: %d\n", rows(), cols(), b);
}

void BDMatrix::printFullStats() {
	printStats();
	for (int i=0; i<blockNum(); i++)
		printBlockStats(i);
}

void BDMatrix::printBlockStats(int index) {
	printf("\tblock %d, SzTot=%.1f dim: %dx%d\n", index, blockValues[index] , (int) blocks[index].rows(), (int) blocks[index].cols());
}


//int BDMatrix::dim() {return rows();}

/*int BDMatrix::maxBlockDim() {
	int mbd = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].size() > 0)
			mbd = max(mbd,(int) blocks[i].rows());
	return mbd;
}*/

double BDMatrix::getBlockValue(const int index) {
	return blockValues[index];
}

void BDMatrix::setBlockValue(const int index, double value) {
	blockValues[index] = value;
}

int BDMatrix::getIndexByValue(double value) {
	int index = 0;
	while (index < b && blockValues[index] != value) index++;
	if (index < b) return index;
	return -1;
}

BDMatrix BDMatrix::operator+ (BDMatrix matrix) {
	if (blockNum() != matrix.blockNum())
		throw "Block num doesn't match!";

	BDMatrix newMatrix(blockNum());
	for (int i=0; i<blockNum(); i++) {
		if (matrix[i].rows() != blocks[i].rows())
			throw ("Blocks[i] size doesn't match!");
		newMatrix[i] = blocks[i] + matrix[i];
	}

	return newMatrix;
}

/*BDMatrix BDMatrix::operator* (BDMatrix matrix) {
	if (blockNum() != matrix.blockNum())
		throw "Block num doesn't match!";

	BDMatrix newMatrix(blockNum());
	for (int i=0; i<blockNum(); i++) {
		if (matrix[i].rows() != blocks[i].rows())
			throw ("Blocks[i] size doesn't match!");
		newMatrix[i] = blocks[i] * matrix[i];
	}

	return newMatrix;
}

BDMatrix BDMatrix::operator* (double scalar) {
	BDMatrix newMatrix(blockNum());
	for (int i=0; i<blockNum(); i++)
		newMatrix[i] = scalar*blocks[i];
	return newMatrix;
}

BDMatrix operator* (double scalar, BDMatrix matrix) {
	BDMatrix newMatrix(matrix.blockNum());
	for (int i=0; i<matrix.blockNum(); i++)
		newMatrix[i] = scalar*matrix[i];
	return newMatrix;
}*/

double BDMatrix::norm() {
	double nrm = 0;
	for (int i=0; i<blockNum(); i++)
		nrm += blocks[i].squaredNorm();
	return sqrt(nrm);
}

double BDMatrix::dot(BDMatrix matrix) {
	double dotProd = 0;
	for (int i=0; i<blockNum(); i++)
		for (int j=0; j<blocks[i].rows(); j++)
			for (int k=0; k<blocks[i].cols(); k++)
				dotProd += blocks[i](j,k) * matrix[i](j,k);
	return dotProd;
}

void BDMatrix::operator=(BDMatrix matrix) {
	resize(matrix.blockNum());
	for (int i=0; i<blockNum(); i++) {
		blocks[i] = matrix[i];
		blockValues[i] = matrix.getBlockValue(i);
	}
}

void BDMatrix::resize(int newBlockNum) {
	if (b != newBlockNum) {
		b = newBlockNum;
		blocks.resize(b);
		blockValues.resize(b);
	}
}

VectorXd BDMatrix::flatten() {
	int d = 0, vecInd = 0;
	for (int i=0; i<blockNum(); i++) d += (int) blocks[i].size();
	VectorXd vector(d);
	for (int i=0; i<blockNum(); i++)
		for (int j=0; j<blocks[i].rows(); j++)
			for (int k=0; k<blocks[i].cols(); k++)
				vector(vecInd++) = blocks[i](j,k);
	return vector;
}
