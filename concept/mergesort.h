// mergesort.h
#pragma once

#include "isortfunction.h"

class MergeSort : public ISortFunction {
public:
    void sort(std::vector<std::string>& arr) override;
    
private:
    void merge(std::vector<std::string>& arr, int left, int mid, int right);
    void mergeSort(std::vector<std::string>& arr, int left, int right);
};
