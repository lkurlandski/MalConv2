// iencryptionlibrary.h
#pragma once

#include <fstream>
#include <iostream>
#include <string>

class IEncryptionLibrary {
public:
    virtual void encrypt(std::string& data) = 0;
    virtual void decrypt(std::string& data) = 0;

    // Default implementation for encryptFile and decryptFile
    virtual void encryptFile(const std::string& inputFile, const std::string& outputFile) {
        // Read input file contents
        std::ifstream input(inputFile, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        input.close();

        // Encrypt data
        encrypt(data);

        // Write encrypted data to output file
        std::ofstream output(outputFile, std::ios::binary);
        output << data;
        output.close();
    }

    virtual void decryptFile(const std::string& inputFile, const std::string& outputFile) {
        // Read input file contents
        std::ifstream input(inputFile, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        input.close();

        // Decrypt data
        decrypt(data);

        // Write decrypted data to output file
        std::ofstream output(outputFile, std::ios::binary);
        output << data;
        output.close();
    }

    virtual ~IEncryptionLibrary() {}
};
