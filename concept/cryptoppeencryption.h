// cryptoppeencryption.h
#pragma once

#include "cryptopp/aes.h"
#include "cryptopp/base64.h"
#include "cryptopp/cryptlib.h"
#include "cryptopp/filters.h"
#include "cryptopp/modes.h"
#include "iencryptionlibrary.h"

class CryptoPPEncryption : public IEncryptionLibrary {
public:
    void encrypt(std::string& data) override;
    void decrypt(std::string& data) override;
};
