// opensslencryption.cpp
#include "opensslencryption.h"
#include <openssl/evp.h>
#include <openssl/rand.h>

void OpenSSLEncryption::encrypt(std::string& data) {
    // Generate random key and IV
    unsigned char key[EVP_MAX_KEY_LENGTH];
    unsigned char iv[EVP_MAX_IV_LENGTH];
    RAND_bytes(key, EVP_MAX_KEY_LENGTH);
    RAND_bytes(iv, EVP_MAX_IV_LENGTH);

    // Create cipher context
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv);

    // Encrypt the data
    int len = data.length();
    int ciphertext_len = 0;
    unsigned char ciphertext[len + EVP_MAX_BLOCK_LENGTH];
    EVP_EncryptUpdate(ctx, ciphertext, &ciphertext_len, (unsigned char*)data.c_str(), len);
    int ciphertext_final_len = 0;
    EVP_EncryptFinal_ex(ctx, ciphertext + ciphertext_len, &ciphertext_final_len);
    ciphertext_len += ciphertext_final_len;

    // Set the encrypted data as the result
    data = std::string(reinterpret_cast<char*>(ciphertext), ciphertext_len);

    // Clean up
    EVP_CIPHER_CTX_free(ctx);
}

void OpenSSLEncryption::decrypt(std::string& data) {
    // Create cipher context
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, (unsigned char*)"01234567890123456789012345678901", (unsigned char*)"0123456789012345");

    // Decrypt the data
    int len = data.length();
    int plaintext_len = 0;
    unsigned char plaintext[len + EVP_MAX_BLOCK_LENGTH];
    EVP_DecryptUpdate(ctx, plaintext, &plaintext_len, (unsigned char*)data.c_str(), len);
    int plaintext_final_len = 0;
    EVP_DecryptFinal_ex(ctx, plaintext + plaintext_len, &plaintext_final_len);
    plaintext_len += plaintext_final_len;

    // Set the decrypted data as the result
    data = std::string(reinterpret_cast<char*>(plaintext), plaintext_len);

    // Clean up
    EVP_CIPHER_CTX_free(ctx);
}
