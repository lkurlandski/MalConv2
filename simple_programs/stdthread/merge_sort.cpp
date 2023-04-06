#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

void merge(std::vector<int>& arr, int start, int mid, int end) {
    std::vector<int> temp(end - start + 1);
    int i = start, j = mid+1, k = 0;

    while (i <= mid && j <= end) {
        if (arr[i] < arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= end) temp[k++] = arr[j++];

    std::copy(temp.begin(), temp.end(), arr.begin()+start);
}

void mergeSort(std::vector<int>& arr, int start, int end, int threads) {
    if (start < end) {
        int mid = (start + end) / 2;

        if (threads > 1) {
            std::thread left(mergeSort, std::ref(arr), start, mid, threads/2);
            std::thread right(mergeSort, std::ref(arr), mid+1, end, threads/2);

            left.join();
            right.join();
        }
        else {
            mergeSort(arr, start, mid, 1);
            mergeSort(arr, mid+1, end, 1);
        }

        merge(arr, start, mid, end);
    }
}

void parallelMergeSort(std::vector<int>& arr, int threads) {
    mergeSort(arr, 0, arr.size()-1, threads);
}

int main() {
    std::vector<int> arr = { 10, 3, 4, 1, 8, 2, 7, 5, 6, 9 };

    int threads = std::thread::hardware_concurrency();
    parallelMergeSort(arr, threads);

    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";

    return 0;
}
