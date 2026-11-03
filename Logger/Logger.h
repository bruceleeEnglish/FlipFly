#ifndef SIMILARITY_JOIN_LOGGER_H
#define SIMILARITY_JOIN_LOGGER_H
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

class Logger {
public:
    Logger();
    void SetStartTimer();
    void SetEndTimer();
    double GetDurationTime();

private:
    std::chrono::steady_clock::time_point startTime, endTime;
};

class HPLogger {
public:
    HPLogger();
    void SetStartTimer();
    void SetEndTimer();
    double GetDurationTime();

private:
    std::chrono::high_resolution_clock::time_point startTime, endTime;
};


#endif //SIMILARITY_JOIN_LOGGER_H
