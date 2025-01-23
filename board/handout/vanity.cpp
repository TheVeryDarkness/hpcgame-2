#include <iostream>  
#include <iomanip>  
#include <sstream>  
#include <cstring>  
#include <fstream>  
#include <random>  
#include <openssl/sha.h>  
#include <openssl/evp.h>  
#include <secp256k1.h>  
  
std::string toHex(const uint8_t* data, size_t size) {  
    std::ostringstream oss;  
    for (size_t i = 0; i < size; ++i) {  
        oss << std::hex << std::setfill('0') << std::setw(2) << (int)data[i];  
    }  
    return oss.str();  
}  
   
std::string sha3256(const uint8_t* data, size_t size) {  
    EVP_MD_CTX* context = EVP_MD_CTX_new();  
    const EVP_MD* md = EVP_get_digestbyname("sha3-256");  
    EVP_DigestInit_ex(context, md, nullptr);  
    EVP_DigestUpdate(context, data, size);  
  
    uint8_t hash[EVP_MAX_MD_SIZE];  
    unsigned int hashLen;  
    EVP_DigestFinal_ex(context, hash, &hashLen);  
    EVP_MD_CTX_free(context);  
  
    return toHex(hash, hashLen);  
}  

void generateRandomPrivateKey(uint8_t privateKey[32], std::mt19937_64 &gen) {  
    for (uint_fast64_t *p = (uint_fast64_t *)privateKey; p < (uint_fast64_t *)(privateKey + 32); ++p) {
        *p = gen();
    }
    // FILE* urandom = fopen("/dev/urandom", "rb");  
    // int res = fread(privateKey, 1, 32, urandom);  
    // if (res != 32) {  
    //     std::cerr << "Failed to read random data" << std::endl;  
    //     exit(1);  
    // }
    // fclose(urandom);  
}  

std::string computeEthereumAddress(const secp256k1_context* ctx, const uint8_t privateKey[32]) {  
    secp256k1_pubkey pubkey;  
    secp256k1_ec_pubkey_create(ctx, &pubkey, privateKey);
    uint8_t pubkeySerialized[65];  
    size_t pubkeySerializedLen = 65;  
    secp256k1_ec_pubkey_serialize(ctx, pubkeySerialized, &pubkeySerializedLen, &pubkey, SECP256K1_EC_UNCOMPRESSED);  
  
    std::string hash = sha3256(pubkeySerialized + 1, pubkeySerializedLen - 1);  

    return "0x" + hash.substr(24);  
}  
  
int main(int argc, char* argv[]) {
    std::ifstream infile("vanity.in");
    std::ofstream outfile("vanity.out");

    std::string vanityPrefixes[10];
    uint8_t privateKeys[10][32];
    std::string addresses[10];

    for (int i = 0; i < 10; ++i) {
        infile >> vanityPrefixes[i];
    }
    std::random_device rd;

    #pragma omp parallel for
    for(int i = 0; i < 10; ++i){
        std::mt19937_64 gen(rd());

        secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN); 

        const std::string &vanityPrefix = vanityPrefixes[i];
        uint8_t* const privateKey = privateKeys[i];
        std::string &address = addresses[i];

        while (true) {  
            generateRandomPrivateKey(privateKey, gen);  
            address = computeEthereumAddress(ctx, privateKey);  
            if (address.substr(2, vanityPrefix.size()) == vanityPrefix) {  
                break;  
            }  
        }  
        secp256k1_context_destroy(ctx);  
    }
    for (int i = 0; i < 10; ++i) {
        const uint8_t* const privateKey = privateKeys[i];
        std::string &address = addresses[i];

        outfile << address << std::endl;  
        outfile << toHex(privateKey, 32) << std::endl;  
    }
    return 0;  
}  