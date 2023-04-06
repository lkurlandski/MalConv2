// quicksort.h
#pragma once

#include "isortfunction.h"

class QuickSort : public ISortFunction {
public:
    void sort(std::vector<std::string>& arr) override;

private:
    void quickSort(std::vector<std::string>& arr, int low, int high);
    int partition(std::vector<std::string>& arr, int low, int high);
};
