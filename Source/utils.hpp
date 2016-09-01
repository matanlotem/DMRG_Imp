/*
 * utils.hpp
 *
 *  Created on: Aug 29, 2016
 *      Author: Matan
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <Eigen/Core>
using namespace Eigen;

/*class utils {
public:
	static MatrixXd tensorProdIdent(MatrixXd matrixA, int identDim);
	static MatrixXd tensorProdIdent(int identDim, MatrixXd matrixB);
};*/
namespace utils {
	MatrixXd tensorProdIdent(MatrixXd matrixA, int identDim);
	MatrixXd tensorProdIdent(int identDim, MatrixXd matrixB);
}

MatrixXd utils::tensorProdIdent(MatrixXd matrixA, int identDim) {
	MatrixXd outputMatrix(matrixA.rows()*identDim, matrixA.cols()*identDim);
	outputMatrix.setZero();
	for (int i=0; i<matrixA.rows(); i++)
		for (int j=0; j<matrixA.cols(); j++)
			for (int k=0; k<identDim; k++)
				outputMatrix(i*identDim + k,j*identDim + k) = matrixA(i,j);
	return outputMatrix;
}

MatrixXd utils::tensorProdIdent(int identDim, MatrixXd matrixB) {
	MatrixXd outputMatrix(matrixB.rows()*identDim, matrixB.cols()*identDim);
	outputMatrix.setZero();
	for (int k=0; k<identDim; k++)
		for (int i=0; i<matrixB.rows(); i++)
			for (int j=0; j<matrixB.cols(); j++)
				outputMatrix(k*matrixB.rows() + i, k*matrixB.cols() + j) = matrixB(i,j);
	return outputMatrix;
}



#endif /* UTILS_HPP_ */
