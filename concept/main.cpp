#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include "ISortFunction.h"
#include "mergesort.h"
// #include "quicksort.h"
// #include "heapsort.h"
#include "IEncryptionLibrary.h"
#include "OpenSSLEncryption.h"
// #include "CryptoPPEncryption.h"
// #include "TomCryptEncryption.h"

void getAllFiles(const std::string& directory, std::vector<std::string>& files) {
    DIR* dir;
    struct dirent* entry;

    if ((dir = opendir(directory.c_str())) == nullptr) {
        return;
    }

    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName = entry->d_name;
        std::string filePath = directory + "/" + fileName;

        if (entry->d_type == DT_REG) {  // Check if it's a regular file
            files.push_back(filePath);
        } else if (entry->d_type == DT_DIR && fileName != "." && fileName != "..") {  // Check if it's a directory
            getAllFiles(filePath, files);  // Recursively search for files in subdirectories
        }
    }

    closedir(dir);
}

void sortAndEncryptFiles(const std::string& rootPath, ISortFunction* sortFunction, IEncryptionLibrary* encryptionLibrary) {
    // Find files in root path
    std::vector<std::string> files;
    getAllFiles(rootPath, files);

    // Sort files using the specified sorting function
    sortFunction->sort(files);

    // Encrypt files using the specified encryption library
    encryptionLibrary->encryptFiles(files);

    // Decrypt files using the specified encryption library
    encryptionLibrary->decryptFiles(files);

    // TODO: Perform further operations on the files as needed
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
