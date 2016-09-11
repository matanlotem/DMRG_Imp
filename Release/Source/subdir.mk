################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Source/SBDMatrix.cpp \
../Source/SBDODMatrix.cpp \
../Source/main.cpp 

OBJS += \
./Source/SBDMatrix.o \
./Source/SBDODMatrix.o \
./Source/main.o 

CPP_DEPS += \
./Source/SBDMatrix.d \
./Source/SBDODMatrix.d \
./Source/main.d 


# Each subdirectory must supply rules for building sources it contributes
Source/%.o: ../Source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


