// cryptoppeencryption.cpp
#include "cryptoppeencryption.h"
#include <cryptopp/hex.h>
#include <cryptopp/osrng.h>

void CryptoPPEncryption::encrypt(std::string& data) {
    // Generate random key and IV
    byte key[CryptoPP::AES::DEFAULT_KEYLENGTH];
    byte iv[CryptoPP::AES::BLOCKSIZE];
    CryptoPP::AutoSeededRandomPool prng;
    prng.GenerateBlock(key, sizeof(key));
    prng.GenerateBlock(iv, sizeof(iv));

    // Encrypt the data
    std::string ciphertext;
    CryptoPP::AES::Encryption aesEncryption(key, sizeof(key));
    CryptoPP::CBC_Mode_ExternalCipher::Encryption cbcEncryption(aesEncryption, iv);
    CryptoPP::StreamTransformationFilter stfEncryptor(cbcEncryption, new CryptoPP::StringSink(ciphertext));
    stfEncryptor.Put(reinterpret_cast<const byte*>(data.c_str()), data.length() + 1);
    stfEncryptor.MessageEnd();

    // Set the encrypted data as the result
    data = ciphertext;
}

void CryptoPPEncryption::decrypt(std::string& data) {
    // Decrypt the data
    std::string decryptedtext;
    CryptoPP::AutoSeededRandomPool prng;
    CryptoPP::StringSource s(reinterpret_cast<const byte*>(data.c_str()), data.size(), true,
        new CryptoPP::HexDecoder(new CryptoPP::StreamTransformationFilter(
            CryptoPP::AES::Decryption("01234567890123456789012345678901", sizeof("01234567890123456789012345678901")),
            new CryptoPP::StringSink(decryptedtext)
        ))
    );

    // Set the decrypted data as the result
    data = decryptedtext;
}
