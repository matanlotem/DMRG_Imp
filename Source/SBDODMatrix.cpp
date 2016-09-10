/*
 * SBDODMatrix.cpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#include "SBDMatrix.hpp"
#include "SBDODMatrix.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

SBDODMatrix::SBDODMatrix(int b) : b(b) {
	blocks.resize(b);
	for (int i=0; i<b; i++) blocks[i] = MatrixXd(0,0);
}

SBDODMatrix::SBDODMatrix(SBDMatrix *matrix) {
	resize(matrix->blockNum()-1);
	for (int i=0; i<b; i++)
		blocks[i] = MatrixXd((*matrix)[i].rows(),(*matrix)[i+1].cols());
}

SBDODMatrix::~SBDODMatrix() {
	//delete[] blocks;
}

void SBDODMatrix::operator=(SBDODMatrix matrix) {
	resize(matrix.blockNum());
	for (int i=0; i<blockNum(); i++)
		blocks[i] = matrix[i];
}

void SBDODMatrix::resize(int newBlockNum) {
	if (b != newBlockNum) {
		b = newBlockNum;
		blocks.resize(b);
	}
}

void SBDODMatrix::resize(SBDMatrix *matrix) {
	resize(matrix->blockNum()-1);
	for (int i=0; i<b; i++)
		blocks[i] = MatrixXd((*matrix)[i].rows(),(*matrix)[i+1].cols());
}


int SBDODMatrix::blockNum() {return b;}

int SBDODMatrix::rows() {
	int d = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].rows() > 0)
			d += blocks[i].rows();
	d += blocks[b-1].cols();
	return d;
}

int SBDODMatrix::cols() {
	int d = blocks[0].rows();
	for (int i=0; i<b; i++)
		if (blocks[i].cols() > 0)
			d += blocks[i].cols();
	return d;
}

MatrixXd& SBDODMatrix::operator[] (const int index) {
	return blocks[index];
}

MatrixXd SBDODMatrix::toMatrix() {
	MatrixXd matrix(rows(),cols());
	matrix.setZero();
	int indRows = 0, indCols = blocks[0].rows();
	for (int i=0; i<blockNum(); i++) {
		matrix.block(indRows,indCols,blocks[i].rows(),blocks[i].cols()) = blocks[i];
		indRows += blocks[i].rows();
		indCols += blocks[i].cols();
	}
	return matrix;
}

void SBDODMatrix::print() {
	cout << toMatrix() << endl;
}

void SBDODMatrix::printStats() {
	printf("dim: %dx%d, blocks: %d\n", rows(), cols(), b);
}

void SBDODMatrix::printFullStats() {
	printStats();
	for (int i=0; i<blockNum(); i++)
		printBlockStats(i);
}


void SBDODMatrix::printBlockStats(int index) {
	printf("\tblock %d dim: %dx%d\n", index, (int) blocks[index].rows(), (int) blocks[index].cols());
}



int SBDODMatrix::dim() {return rows();}

