// cryptoppeencryption.cpp
#include "cryptopp/aes.h"
#include "cryptopp/base64.h"
#include "cryptopp/cryptlib.h"
#include "cryptopp/filters.h"
#include "cryptopp/modes.h"
#include "cryptopp/osrng.h"
#include "cryptoppeencryption.h"

void CryptoPPEncryption::encrypt(std::string& data) {
    // Generate a random key and IV
    CryptoPP::SecByteBlock key(CryptoPP::AES::MAX_KEYLENGTH);
    CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE];
    CryptoPP::AutoSeededRandomPool prng;
    prng.GenerateBlock(key, key.size());
    prng.GenerateBlock(iv, sizeof(iv));

    // Encrypt the data
    CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption encryptor(key, key.size(), iv);
    CryptoPP::StringSource encryptSource(data, true,
        new CryptoPP::StreamTransformationFilter(encryptor,
            new CryptoPP::StringSink(data)));

    // Base64 encode the encrypted data
    CryptoPP::StringSource base64Encoder(data, true,
        new CryptoPP::Base64Encoder(new CryptoPP::StringSink(data), false));
}

void CryptoPPEncryption::decrypt(std::string& data) {
    // Base64 decode the input data
    CryptoPP::StringSource base64Decoder(data, true,
        new CryptoPP::Base64Decoder(new CryptoPP::StringSink(data)));

    // Decrypt the data
    CryptoPP::SecByteBlock key(CryptoPP::AES::MAX_KEYLENGTH);
    CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE];
    CryptoPP::StringSource(data, true,
        new CryptoPP::ArraySink(key, key.size()));
    CryptoPP::StringSource(data, true,
        new CryptoPP::ArraySink(iv, sizeof(iv)));
    CryptoPP::CBC_Mode<CryptoPP::AES>::Decryption decryptor(key, key.size(), iv);
    CryptoPP::StringSource decryptSource(data, true,
        new CryptoPP::StreamTransformationFilter(decryptor,
            new CryptoPP::StringSink(data)));
}