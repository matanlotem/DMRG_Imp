################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Source/BDMatrix.cpp \
../Source/BODMatrix.cpp \
../Source/main.cpp 

OBJS += \
./Source/BDMatrix.o \
./Source/BODMatrix.o \
./Source/main.o 

CPP_DEPS += \
./Source/BDMatrix.d \
./Source/BODMatrix.d \
./Source/main.d 


# Each subdirectory must supply rules for building sources it contributes
Source/%.o: ../Source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


