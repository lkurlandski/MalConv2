// tomcryptencryption.cpp
#include "tomcryptencryption.h"
#include <cstring>

void TomCryptEncryption::encrypt(std::string& data) {
    // Generate random key and IV
    unsigned char key[16];
    unsigned char iv[16];
    int err;

    if ((err = rng_make_prng(128, find_prng("yarrow"), nullptr, &yarrow_prng)) != CRYPT_OK) {
        // Handle error
        return;
    }

    if ((err = rng_get_bytes(yarrow_prng, key, sizeof(key), nullptr)) != CRYPT_OK) {
        // Handle error
        return;
    }

    if ((err = rng_get_bytes(yarrow_prng, iv, sizeof(iv), nullptr)) != CRYPT_OK) {
        // Handle error
        return;
    }

    // Encrypt the data
    symmetric_CBC cbc;
    if ((err = cbc_start(find_cipher("aes"), iv, key, sizeof(key), 0, &cbc)) != CRYPT_OK) {
        // Handle error
        return;
    }

    char* input = const_cast<char*>(data.c_str());
    int input_len = static_cast<int>(data.length());
    int output_len = input_len + (16 - (input_len % 16));
    char* output = new char[output_len];

    if ((err = cbc_encrypt(reinterpret_cast<const unsigned char*>(input), reinterpret_cast<unsigned char*>(output),
        output_len, &cbc)) != CRYPT_OK) {
        // Handle error
        delete[] output;
        return;
    }

    // Set the encrypted data as the result
    data.assign(output, output_len);
    delete[] output;
}

void TomCryptEncryption::decrypt(std::string& data) {
    // Decrypt the data
    symmetric_CBC cbc;
    unsigned char key[16];
    unsigned char iv[16];
    int err;

    if ((err = rng_make_prng(128, find_prng("yarrow"), nullptr, &yarrow_prng)) != CRYPT_OK) {
        // Handle error
        return;
    }

    if ((err = rng_get_bytes(yarrow_prng, key, sizeof(key), nullptr)) != CRYPT_OK) {
        // Handle error
        return;
    }

    if ((err = rng_get_bytes(yarrow_prng, iv, sizeof(iv), nullptr)) != CRYPT_OK) {
        // Handle error
        return;
    }

    if ((err = cbc_start(find_cipher("aes"), iv, key, sizeof(key), 0, &cbc)) != CRYPT_OK) {
        // Handle error
        return;
    }

    char* input = const_cast<char*>(data.c_str());
    int input_len = static_cast<int>(data.length());
    char* output = new char[input_len];

    if ((err = cbc_decrypt(reinterpret_cast<const unsigned char*>(input), reinterpret_cast<unsigned char*>(output),
        input_len, &cbc)) != CRYPT_OK) {
        // Handle error
        delete[] output;
        return;
    }

    // Set the decrypted data as the result
    data.assign(output, input_len);
    delete[] output;
}
