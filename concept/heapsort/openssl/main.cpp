#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include "..\..\isortfunction.h"
#include "..\..\heapsort.h"
#include "..\..\iencryptionlibrary.h"
#include "..\..\opensslencryption.h"

#ifdef _WIN32
#include <direct.h> // for _mkdir on Windows
#else
#include <sys/stat.h> // for mkdir on Unix-like systems
#endif


void createDirectory(const std::string& directoryName) {
    // Create a directory
#ifdef _WIN32
    // Windows-specific code to create directory
    int result = _mkdir(directoryName.c_str());
#else
    // Unix-like systems
    int result = mkdir(directoryName.c_str(), 0777);
#endif

    if (result == 0) {
        std::cout << "Created directory: " << directoryName << std::endl;
    } else {
        std::cerr << "Failed to create directory: " << directoryName << std::endl;
    }
}


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
            std::string fullPath = dirPath + "\\" + entryName;
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

    createDirectory("encrypted");
    createDirectory("decrypted");

    // Encrypt files using the specified encryption library
    for (const std::string& file : files) {
        std::string encryptedFile = "encrypted\\" + file.substr(file.find_last_of("/\\") + 1) + ".enc"; // Encrypted file path
        std::cout << "Encrypting file: " << file << " into " << encryptedFile << std::endl;
        encryptionLibrary->encryptFile(file, encryptedFile); // Encrypt the file content
    }

    files = getAllFiles("encrypted");
    // Decrypt files using the specified encryption library
    for (const std::string& file : files) {
        std::string decryptedFile = "decrypted\\" + file.substr(file.find_last_of("/\\") + 1) + ".dec"; // Decrypted file path
        std::cout << "Decrypting file: " << file << " into " << decryptedFile << std::endl;
        encryptionLibrary->decryptFile(file, decryptedFile); // Decrypt the file content
    }

}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << std::endl;
        return 1;
    }

    std::string rootPath = argv[1];

    // Create an instance of the specified sorting function
    ISortFunction* sortFunction = new HeapSort();

    // Create an instance of the OpenSSL encryption library
    IEncryptionLibrary* encryptionLibrary = new OpenSSLEncryption();

    // Call the sortAndEncryptFiles function with the specified root path, sorting function, and encryption library
    sortAndEncryptFiles(rootPath, sortFunction, encryptionLibrary);

    // Clean up memory
    delete sortFunction;
    delete encryptionLibrary;

    return 0;
}
