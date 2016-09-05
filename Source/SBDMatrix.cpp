/*
 * SBDMatrix.cpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#include "SBDMatrix.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

SBDMatrix::SBDMatrix(int b) : b(b) {
	blocks = new MatrixXd[b];
	blockValues = new double[b];
	for (int i=0; i<b; i++) blocks[i] = MatrixXd(0,0);
}

SBDMatrix::SBDMatrix(SBDMatrix * matrix, VectorXd * vector) {
	printf("a\n");
	b = matrix->blockNum();
	blocks = new MatrixXd[b];
	blockValues = new double[b];
	printf("b\n");
	int ind=0, rows, cols;
	for (int i=0; i<b; i++){
		printf("\t%d\n",i);
		rows = (*matrix)[i].rows();
		cols = (*matrix)[i].cols();

		blocks[i] = MatrixXd(rows,cols);
		for (int j=0; j<rows; j++)
			for (int k=0; k<cols; k++)
				blocks[i](j,k) = (*vector)(j*rows+k+ind);
		ind+=rows*cols;
	}
	printf("c\n");
	if (ind != matrix->rows() * matrix->cols()) {
		printf("ind=%d  matrix: %dx%d\n",ind, matrix->rows(), matrix->cols());
		throw "matrix-vector dimensions don't match!";
		abort();
	}
	printf("d\n");

}

SBDMatrix::~SBDMatrix() {
	printf("Destructor %d started\n",this);
	delete[] blocks;
	delete[] blockValues;
	printf("Destructor %d done\n",this);
}

int SBDMatrix::blockNum() {return b;}

int SBDMatrix::rows() {
	int d = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].rows() > 0)
			d += blocks[i].rows();
	return d;
}

int SBDMatrix::cols() {
	int d = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].cols() > 0)
			d += blocks[i].cols();
	return d;
}

MatrixXd& SBDMatrix::operator[] (const int index) {
	return blocks[index];
}

MatrixXd SBDMatrix::toMatrix() {
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

void SBDMatrix::print() {
	cout << toMatrix() << endl;
}

void SBDMatrix::printStats() {
	printf("dim: %dx%d, blocks: %d\n", rows(), cols(), b);
}

void SBDMatrix::printBlockStats(int index) {
	printf("\tblock %d dim: %dx%d\n", index, (int) blocks[index].rows(), (int) blocks[index].cols());
}


int SBDMatrix::dim() {return rows();}

int SBDMatrix::maxBlockDim() {
	int mbd = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].size() > 0)
			mbd = max(mbd,(int) blocks[i].rows());
	return mbd;
}

double SBDMatrix::getBlockValue(const int index) {
	return blockValues[index];
}

void SBDMatrix::setBlockValue(const int index, double value) {
	blockValues[index] = value;
}

int SBDMatrix::getIndexByValue(double value) {
	int index = 0;
	while (index < b && blockValues[index] != value) index++;
	if (index < b) return index;
	return -1;
}

SBDMatrix SBDMatrix::operator+ (SBDMatrix matrix) {
	if (blockNum() != matrix.blockNum())
		throw "Block num doesn't match!";

	SBDMatrix newMatrix(blockNum());
	for (int i=0; i<blockNum(); i++) {
		if (matrix[i].rows() != blocks[i].rows())
			throw ("Blocks[i] size doesn't match!");
		newMatrix[i] = blocks[i] + matrix[i];
	}

	return newMatrix;
}

SBDMatrix SBDMatrix::operator* (SBDMatrix matrix) {
	if (blockNum() != matrix.blockNum())
		throw "Block num doesn't match!";

	SBDMatrix newMatrix(blockNum());
	for (int i=0; i<blockNum(); i++) {
		if (matrix[i].rows() != blocks[i].rows())
			throw ("Blocks[i] size doesn't match!");
		newMatrix[i] = blocks[i] * matrix[i];
	}

	return newMatrix;
}

SBDMatrix SBDMatrix::operator* (double scalar) {
	SBDMatrix newMatrix(blockNum());
	for (int i=0; i<blockNum(); i++)
		newMatrix[i] = scalar*blocks[i];
	return newMatrix;
}

SBDMatrix operator* (double scalar, SBDMatrix matrix) {
	SBDMatrix newMatrix(matrix.blockNum());
	for (int i=0; i<matrix.blockNum(); i++)
		newMatrix[i] = scalar*matrix[i];
	return newMatrix;
}

double SBDMatrix::norm() {
	double nrm = 0;
	for (int i=0; i<blockNum(); i++)
		nrm += blocks[i].squaredNorm();
	return sqrt(nrm);
}

double SBDMatrix::dot(SBDMatrix matrix) {
	double dotProd = 0;
	for (int i=0; i<blockNum(); i++)
		for (int j=0; j<blocks[i].rows(); j++)
			for (int k=0; k<blocks[i].cols(); k++)
				dotProd += blocks[i](j,k) * matrix[i](j,k);
	return dotProd;
}

void SBDMatrix::operator=(SBDMatrix matrix) {
	if (b != matrix.blockNum()) {
		b = matrix.blockNum();
		delete[] blocks;
		delete[] blockValues;
		blocks = new MatrixXd[b];
		blockValues = new double[b];
	}
	for (int i=0; i<blockNum(); i++) {
		blocks[i] = matrix[i];
		blockValues[i] = matrix.getBlockValue(i);
	}
}


