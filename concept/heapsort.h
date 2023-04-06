//heapsort.h
#pragma once

#include "isortfunction.h"

class HeapSort : public ISortFunction {
public:
    void sort(std::vector<std::string>& arr) override;

private:
    void heapify(std::vector<std::string>& arr, int n, int i);
    void buildHeap(std::vector<std::string>& arr, int n);
};
