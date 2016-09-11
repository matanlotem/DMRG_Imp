/*
 * BDMatrix.hpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#ifndef BDMATRIX_HPP_
#define BDMATRIX_HPP_

#include <vector>
#include <Eigen/Core>
using namespace Eigen;

class BDMatrix {
private:
	int b;
	std::vector<double> blockValues;

public:
	std::vector<MatrixXd> blocks;
	BDMatrix(int b);
	BDMatrix(BDMatrix * matrix, VectorXd * vector);
	~BDMatrix();

	int blockNum();
	int rows();
	int cols();
	MatrixXd& operator[] (const int index);
	MatrixXd toMatrix();
	void print();
	void printStats();
	void printFullStats();
	void printBlockStats(const int index);

	//int dim();
	//int maxBlockDim();

	double getBlockValue(const int index);
	void setBlockValue(const int index, double value);
	int getIndexByValue(double value);

	double norm();
	double dot(BDMatrix matrix);
	void operator=(BDMatrix matrix);
	void resize(int newBlockNum);
	BDMatrix operator+ (BDMatrix matrix);
	BDMatrix operator- (BDMatrix matrix);

	/*BDMatrix operator* (BDMatrix matrix);
	BDMatrix operator* (double scalar);
	BDMatrix operator/ (BDMatrix matrix);
	BDMatrix operator/ (double scalar);*/

	VectorXd flatten();
};

//BDMatrix operator* (double scalar, BDMatrix matrix);

#endif /* BDMATRIX_HPP_ */
