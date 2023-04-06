#include <iostream>
#include <vector>
#include <thread>

void max_heapify(std::vector<int>& v, int i, int n) {
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    int largest = i;

    if (left < n && v[left] > v[largest]) {
        largest = left;
    }

    if (right < n && v[right] > v[largest]) {
        largest = right;
    }

    if (largest != i) {
        std::swap(v[i], v[largest]);
        max_heapify(v, largest, n);
    }
}

void build_max_heap(std::vector<int>& v, int n, int p) {
    int chunk_size = n / p;
    int start = p * chunk_size;
    int end = (p == (p - 1)) ? n : start + chunk_size;

    for (int i = end / 2 - 1; i >= start; i--) {
        max_heapify(v, i, end);
    }
}

void heap_sort(std::vector<int>& v, int n, int p) {
    std::vector<std::thread> threads(p);

    for (int i = 0; i < p; i++) {
        threads[i] = std::thread(build_max_heap, std::ref(v), n, i);
    }

    for (int i = 0; i < p; i++) {
        threads[i].join();
    }

    for (int i = n - 1; i >= 1; i--) {
        std::swap(v[0], v[i]);
        max_heapify(v, 0, i);
    }
}

int main() {
    std::vector<int> v = {7, 3, 8, 2, 1, 9, 4, 6, 5, 0};
    int n = v.size();
    int p = 4;

    heap_sort(v, n, p);

    for (int i = 0; i < n; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
