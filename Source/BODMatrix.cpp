/*
 * BODMatrix.cpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#include "BODMatrix.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>
#include "BDMatrix.hpp"

using namespace Eigen;
using namespace std;

BODMatrix::BODMatrix(int b) : b(b) {
	blocks.resize(b);
	for (int i=0; i<b; i++) blocks[i] = MatrixXd(0,0);
}

BODMatrix::BODMatrix(BDMatrix *matrix) {
	resize(matrix->blockNum()-1);
	for (int i=0; i<b; i++)
		blocks[i] = MatrixXd((*matrix)[i].rows(),(*matrix)[i+1].cols());
}

BODMatrix::~BODMatrix() {
	//delete[] blocks;
}

void BODMatrix::operator=(BODMatrix matrix) {
	resize(matrix.blockNum());
	for (int i=0; i<blockNum(); i++)
		blocks[i] = matrix[i];
}

void BODMatrix::resize(int newBlockNum) {
	if (b != newBlockNum) {
		b = newBlockNum;
		blocks.resize(b);
	}
}

void BODMatrix::resize(BDMatrix *matrix) {
	resize(matrix->blockNum()-1);
	for (int i=0; i<b; i++)
		blocks[i] = MatrixXd((*matrix)[i].rows(),(*matrix)[i+1].cols());
}


int BODMatrix::blockNum() {return b;}

int BODMatrix::rows() {
	int d = 0;
	for (int i=0; i<b; i++)
		if (blocks[i].rows() > 0)
			d += blocks[i].rows();
	d += blocks[b-1].cols();
	return d;
}

int BODMatrix::cols() {
	int d = blocks[0].rows();
	for (int i=0; i<b; i++)
		if (blocks[i].cols() > 0)
			d += blocks[i].cols();
	return d;
}

MatrixXd& BODMatrix::operator[] (const int index) {
	return blocks[index];
}

MatrixXd BODMatrix::toMatrix() {
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

void BODMatrix::print() {
	cout << toMatrix() << endl;
}

void BODMatrix::printStats() {
	printf("dim: %dx%d, blocks: %d\n", rows(), cols(), b);
}

void BODMatrix::printFullStats() {
	printStats();
	for (int i=0; i<blockNum(); i++)
		printBlockStats(i);
}


void BODMatrix::printBlockStats(int index) {
	printf("\tblock %d dim: %dx%d\n", index, (int) blocks[index].rows(), (int) blocks[index].cols());
}



int BODMatrix::dim() {return rows();}

