/*
 * BODMatrix.hpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#ifndef BODMATRIX_HPP_
#define BODMATRIX_HPP_

#include <Eigen/Core>
#include <vector>

#include "BDMatrix.hpp"
using namespace Eigen;

class BODMatrix {
private:
	int b;
	std::vector<MatrixXd> blocks;

public:
	BODMatrix(int b);
	BODMatrix(BDMatrix *matrix);
	~BODMatrix();

	int blockNum();
	int dim();
	int rows();
	int cols();
	MatrixXd& operator[] (const int index);
	void operator=(BODMatrix matrix);
	void resize(int newBlockNum);
	void resize(BDMatrix *matrix);

	MatrixXd toMatrix();
	void print();
	void printStats();
	void printFullStats();
	void printBlockStats(const int index);
};





#endif /* BODMATRIX_HPP_ */
