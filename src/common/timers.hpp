#pragma once

#include <iostream>
#include <string>
#include <chrono>

/**
 * @brief High-resolution timer for benchmarking code sections.
 *
 * The Timer class helps measure and report the execution
 * time of arbitrary code blocks in milliseconds. It is used
 * throughout to profile kernels and guide performance tuning.
 */

class Timer {
	
public:
	void start() {
		begin_ = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		auto end = std::chrono::high_resolution_clock::now();
		elapsed_ms_ = std::chrono::duration<double, std::milli>(end - begin_).count();
	}

	double milliseconds() const {
		return elapsed_ms_;
	}

	void print(const std::string& label) const {
		std::cout << "[TIMER] " << label << ": " << elapsed_ms_ << " ms" << std::endl;
	}

private:
	std::chrono::high_resolution_clock::time_point begin_;
	double elapsed_ms_ = 0.0;
};

