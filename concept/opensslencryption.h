// opensslencryption.h
#pragma once

#include "iencryptionlibrary.h"

class OpenSSLEncryption : public IEncryptionLibrary {
public:
    void encrypt(std::string& data) override;
    void decrypt(std::string& data) override;
};

