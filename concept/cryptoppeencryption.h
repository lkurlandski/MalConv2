// cryptoppeencryption.h
#pragma once

#include "iencryptionlibrary.h"
#include <cryptopp/aes.h>
#include <cryptopp/cryptlib.h>
#include <cryptopp/filters.h>
#include <cryptopp/modes.h>

class CryptoPPEncryption : public IEncryptionLibrary {
public:
    void encrypt(std::string& data) override;
    void decrypt(std::string& data) override;
};

