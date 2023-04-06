// iencryptionlibrary.h
#pragma once

#include <string>

class IEncryptionLibrary {
public:
    virtual void encrypt(std::string& data) = 0;
    virtual void decrypt(std::string& data) = 0;
    virtual ~IEncryptionLibrary() {}
};
