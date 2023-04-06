// tomcryptencryption.h
#pragma once

#include "iencryptionlibrary.h"
#include <tomcrypt.h>

class TomCryptEncryption : public IEncryptionLibrary {
public:
    void encrypt(std::string& data) override;
    void decrypt(std::string& data) override;
};

