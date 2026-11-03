#include "Logger.h"

Logger::Logger() {
    startTime = std::chrono::steady_clock::now();
}

void Logger::SetStartTimer() {
    startTime = std::chrono::steady_clock::now();
}

void Logger::SetEndTimer() {
    endTime = std::chrono::steady_clock::now();
}


double Logger::GetDurationTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}

HPLogger::HPLogger() {
    startTime = std::chrono::high_resolution_clock::now();
}

void HPLogger::SetStartTimer() {
    startTime = std::chrono::high_resolution_clock::now();
}

void HPLogger::SetEndTimer() {
    endTime = std::chrono::high_resolution_clock::now();
}

double HPLogger::GetDurationTime() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1e6;
}