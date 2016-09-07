/*
 * SBDODMatrix.hpp
 *
 *  Created on: Sep 1, 2016
 *      Author: Matan
 */

#ifndef SBDODMATRIX_HPP_
#define SBDODMATRIX_HPP_

#include "SBDMatrix.hpp"
#include <Eigen/Core>
#include <vector>
using namespace Eigen;

class SBDODMatrix {
private:
	int b;
	std::vector<MatrixXd> blocks;

public:
	SBDODMatrix(int b);
	SBDODMatrix(SBDMatrix *matrix);
	~SBDODMatrix();

	int blockNum();
	int dim();
	int rows();
	int cols();
	MatrixXd& operator[] (const int index);
	void operator=(SBDODMatrix matrix);
	void resize(int newBlockNum);
	void resize(SBDMatrix *matrix);

	MatrixXd toMatrix();
	void print();
	void printStats();
	void printBlockStats(const int index);
};





#endif /* SBDODMATRIX_HPP_ */
