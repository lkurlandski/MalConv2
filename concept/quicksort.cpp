// quicksort.cpp
#include "quicksort.h"

void QuickSort::sort(std::vector<std::string>& arr) {
    quickSort(arr, 0, arr.size() - 1);
}

void QuickSort::quickSort(std::vector<std::string>& arr, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);
        quickSort(arr, low, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, high);
    }
}

int QuickSort::partition(std::vector<std::string>& arr, int low, int high) {
    std::string pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}
