#include <atomic>  
#include <iostream>  
#include <iomanip>  
#include <sstream>  
#include <cstring>  
#include <fstream>  
#include <random>  
#include <mutex>
#include <shared_mutex>
#include <openssl/sha.h>  
#include <openssl/evp.h>  
#include <secp256k1.h>  

using engine = std::ranlux48_base;
  
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

void generateRandomPrivateKey(uint8_t privateKey[32], engine &gen) {  
    for (uint8_t *p = (uint8_t *)privateKey; p < privateKey + 32; p += 6) {
        const uint_fast64_t r = { gen() };
        p[0] = r >> 40;
        p[1] = r >> 32;
        p[2] = r >> 24;
        p[3] = r >> 16;
        p[4] = r >> 8;
        p[5] = r;
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

constexpr int MAX_THREADS = 8;
constexpr int MAX_VANITY_LENGTH = 10;
  
std::random_device rd;
std::string vanityPrefixes[MAX_VANITY_LENGTH];
uint8_t privateKeys[MAX_VANITY_LENGTH][32];
std::string addresses[MAX_VANITY_LENGTH];
std::mutex mtx;
std::atomic_bool found[MAX_VANITY_LENGTH] = {};

void *run(void *arg) {
    const int init = *(int *)arg;
    std::clog << "Started thread " << init << '\n';
    engine gen(rd() + init); // Avoiding seed collision

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN); 

    uint8_t privateKey[32];

    while (true) {  
        generateRandomPrivateKey(privateKey, gen);  
        const std::string address = computeEthereumAddress(ctx, privateKey);  
        const std::string_view addressView(address);

        size_t found_total = 0;

        for (int j = 0; j < MAX_VANITY_LENGTH; ++j) {
            if (found[j].load()) {
                ++found_total;
                continue;
            }
            const std::string &vanityPrefix = vanityPrefixes[j];
            const std::string_view addressView2 = addressView.substr(2, vanityPrefix.size());

            if (addressView2 == vanityPrefix) {  
                std::clog << "Thread " << init << " found vanity " << j << '\n';
                {
                    found[j].store(true);
                    std::unique_lock lock(mtx);
                    addresses[j] = address;
                    memcpy(privateKeys[j], privateKey, sizeof(privateKey));
                }
            }  
        }
        if (found_total == MAX_VANITY_LENGTH)
            break;
    }  
    secp256k1_context_destroy(ctx);  

    std::clog << "Finished thread " << init << '\n';
    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    static_assert(std::atomic_bool::is_always_lock_free);
    std::ifstream infile("vanity.in");
    std::ofstream outfile("vanity.out");

    for (int i = 0; i < MAX_VANITY_LENGTH; ++i) {
        infile >> vanityPrefixes[i];
    }

    pthread_t threads[MAX_THREADS];
    int args[MAX_THREADS];

    for(int i = 0; i < MAX_THREADS; ++i){
        args[i] = i;
        pthread_create(&threads[i], nullptr, run, args + i);
    }

    for(int i = 0; i < MAX_THREADS; ++i){
        pthread_join(threads[i], nullptr);
    }

    for (int i = 0; i < MAX_VANITY_LENGTH; ++i) {
        const uint8_t* const privateKey = privateKeys[i];
        std::string &address = addresses[i];

        outfile << address << std::endl;  
        outfile << toHex(privateKey, 32) << std::endl;  
    }
    return 0;  
}  