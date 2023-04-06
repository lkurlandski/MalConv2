// isortfunction.h
#pragma once

#include <vector>
#include <string>

class ISortFunction {
public:
    virtual void sort(std::vector<std::string>& arr) = 0;
    // Other common interface methods or virtual destructor if needed
};
