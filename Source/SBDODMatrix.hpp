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
using namespace Eigen;

class SBDODMatrix {
private:
	int b;
	MatrixXd * blocks;

public:
	SBDODMatrix(int b);
	SBDODMatrix(SBDMatrix *matrix);
	~SBDODMatrix();

	int blockNum();
	int dim();
	int rows();
	int cols();
	MatrixXd& operator[] (const int index);

	MatrixXd toMatrix();
	void print();
};





#endif /* SBDODMATRIX_HPP_ */
