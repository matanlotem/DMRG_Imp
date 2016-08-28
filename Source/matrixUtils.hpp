/*
 * matrixUtils.hpp
 *
 *  Created on: Aug 24, 2016
 *      Author: Matan
 */

#ifndef MATRIXUTILS_HPP_
#define MATRIXUTILS_HPP_

class matrixUtils{
public:
	template<typename T> static void zero(int dim, T** matrix);
	template<typename T> static void zero(int rows, int cols, T** matrix);
	template<typename T> static void identity(int dim, T** matrix);
	template<typename T> static void copy(int dim, T** matrixA, T** matrixB);
	template<typename T> static void copy(int rows, int cols, T** matrixA, T** matrixB);
	template<typename T> static void truncate(int rowsB, int colsB, T** matrixA, T** matrixB);

	template<typename T> static void transpose(int dim, T** matrixA, T** matrixB);
	template<typename T> static void transpose(int rowsA, int colsA, T** matrixA, T** matrixB);
	template<typename T> static void dagger(int dim, T** matrixA, T** matrixB);
	template<typename T> static void dagger(int rowsA, int colsA, T** matrixA, T** matrixB);
	template<typename T> static void mirror(int dim, T** matrixA, T** matrixB);
};


#endif /* MATRIXUTILS_HPP_ */
