#pragma once

#include <iostream>
#include <cstdlib>
#include <string>

inline void die(const std::string& msg) {
	std::cerr << "[ERROR]" << msg << std::endl;
	std::exit(EXIT_FAILURE);
}

inline void log(const std::string& msg) {
	std::cout << "[LOG]" << msg << std::endl;
}

