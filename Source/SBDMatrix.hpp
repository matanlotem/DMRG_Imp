/*
 * SBDMatrix.hpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#ifndef SBDMATRIX_HPP_
#define SBDMATRIX_HPP_

#include <vector>
#include <Eigen/Core>
using namespace Eigen;

class SBDMatrix {
private:
	int b;
	std::vector<double> blockValues;

public:
	std::vector<MatrixXd> blocks;
	SBDMatrix(int b);
	SBDMatrix(SBDMatrix * matrix, VectorXd * vector);
	~SBDMatrix();

	int blockNum();
	int rows();
	int cols();
	MatrixXd& operator[] (const int index);
	MatrixXd toMatrix();
	void print();
	void printStats();
	void printBlockStats(const int index);

	int dim();
	int maxBlockDim();
	double getBlockValue(const int index);
	void setBlockValue(const int index, double value);
	int getIndexByValue(double value);

	double norm();
	double dot(SBDMatrix matrix);
	void operator=(SBDMatrix matrix);
	void resize(int newBlockNum);
	SBDMatrix operator+ (SBDMatrix matrix);
	SBDMatrix operator- (SBDMatrix matrix);
	SBDMatrix operator* (SBDMatrix matrix);
	SBDMatrix operator* (double scalar);
	SBDMatrix operator/ (SBDMatrix matrix);
	SBDMatrix operator/ (double scalar);


};

SBDMatrix operator* (double scalar, SBDMatrix matrix);

#endif /* SBDMATRIX_HPP_ */
