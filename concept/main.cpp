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


// Function to check if a given path is a directory
bool isDirectory(const std::string& path) {
    struct stat fileStat;
    if (stat(path.c_str(), &fileStat) != 0) {
        return false;
    }
    return S_ISDIR(fileStat.st_mode);
}

// Function to get all files in a directory (excluding subdirectories)
std::vector<std::string> getAllFiles(const std::string& dirPath) {
    std::vector<std::string> files;
    DIR* dir;
    struct dirent* ent;

    if ((dir = opendir(dirPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string entryName = ent->d_name;
            std::string fullPath = dirPath + "/" + entryName;
            if (!isDirectory(fullPath)) { // Filter out directories
                files.push_back(fullPath);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Failed to open directory: " << dirPath << std::endl;
    }

    return files;
}

void sortAndEncryptFiles(const std::string& rootPath, ISortFunction* sortFunction, IEncryptionLibrary* encryptionLibrary) {
    // Find files in root path
    std::vector<std::string> files = getAllFiles(rootPath);

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
