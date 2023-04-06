#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

void quickSort(std::vector<int>& arr, int start, int end) {
    if (start < end) {
        int pivot = arr[end];
        int i = start - 1;

        for (int j = start; j < end; j++) {
            if (arr[j] <= pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i+1], arr[end]);

        std::thread left(quickSort, std::ref(arr), start, i);
        std::thread right(quickSort, std::ref(arr), i+2, end);

        left.join();
        right.join();
    }
}

void parallelQuickSort(std::vector<int>& arr, int threads) {
    quickSort(arr, 0, arr.size()-1);
}

int main() {
    std::vector<int> arr = { 10, 3, 4, 1, 8, 2, 7, 5, 6, 9 };

    int threads = std::thread::hardware_concurrency();
    parallelQuickSort(arr, threads);

    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";

    return 0;
}
