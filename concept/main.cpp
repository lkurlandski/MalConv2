#include <iostream>
#include <string>
#include "ISortFunction.h"
#include "mergesort.h"
// #include "quicksort.h"
// #include "heapsort.h"
#include "IEncryptionLibrary.h"
#include "OpenSSLEncryption.h"
// #include "CryptoPPEncryption.h"
// #include "TomCryptEncryption.h"

void sortAndEncryptFiles(const std::string& rootPath, ISortFunction* sortFunction, IEncryptionLibrary* encryptionLibrary) {
    // Find files in root path

    // Sort files using the specified sorting function
    sortFunction->sort();

    // Encrypt files using the specified encryption library
    encryptionLibrary->encrypt();

    // Decrypt files using the specified encryption library
    encryptionLibrary->decrypt();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <root_path> <sort_type>" << std::endl;
        return 1;
    }

    std::string rootPath = argv[1];
    std::string sortType = argv[2];

    // Create an instance of the specified sorting function
    ISortFunction* sortFunction = new MergeSort();

    // Create an instance of the OpenSSL encryption library
    IEncryptionLibrary* encryptionLibrary = new OpenSSLEncryption();

    // Call the sortAndEncryptFiles function with the specified root path, sorting function, and encryption library
    sortAndEncryptFiles(rootPath, sortFunction, encryptionLibrary);

    // Clean up memory
    delete sortFunction;
    delete encryptionLibrary;

    return 0;
}
