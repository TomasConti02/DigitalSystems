#include <iostream>
#include <fstream>
#include <cstdlib> // per aligned_alloc o posix_memalign
#include <cstring> // per memset
#include <immintrin.h> // Intrinseci di Intel x86
#define SSE_DATA_LANE 16 // Allineamento a 16 byte

struct WAVHeader {
    char chunkID[4];
    uint32_t chunkSize;
    char format[4];
    char subchunk1ID[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char subchunk2ID[4];
    uint32_t subchunk2Size;
};

void print_register(__m128i reg) {
    // Array temporaneo per memorizzare il contenuto del registro
    int16_t temp[8] __attribute__((aligned(SSE_DATA_LANE)));

    // Memorizza il contenuto del registro nell'array temporaneo
    _mm_store_si128((__m128i*)temp, reg);

    // Stampa il contenuto dell'array
    printf("Contenuto del registro: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", temp[i]);
    }
    printf("\n");
}

void printWAVHeader(const WAVHeader& header) {
    std::cout << "chunkID: " << std::string(header.chunkID, 4) << std::endl;
    std::cout << "chunkSize: " << header.chunkSize << std::endl;
    std::cout << "format: " << std::string(header.format, 4) << std::endl;
    std::cout << "subchunk1ID: " << std::string(header.subchunk1ID, 4) << std::endl;
    std::cout << "subchunk1Size: " << header.subchunk1Size << std::endl;
    std::cout << "audioFormat: " << header.audioFormat << std::endl;
    std::cout << "numChannels: " << header.numChannels << std::endl;
    std::cout << "sampleRate: " << header.sampleRate << " Hz" << std::endl;
    std::cout << "byteRate: " << header.byteRate << " bytes/sec" << std::endl;
    std::cout << "blockAlign: " << header.blockAlign << " bytes" << std::endl;
    std::cout << "bitsPerSample: " << header.bitsPerSample << " bits" << std::endl;
    std::cout << "subchunk2ID: " << std::string(header.subchunk2ID, 4) << std::endl;
    std::cout << "subchunk2Size: " << header.subchunk2Size << " bytes" << std::endl;
}

void printSampleValues(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary); //we open the fil in binary mode
    if (!file) {
        std::cerr << "Errore: impossibile aprire il file." << std::endl;
        return;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header)); //header reading
    printWAVHeader(header);
    std::cout << "\nNumero di canali: " << header.numChannels
              << "\nFrequenza di campionamento: " << header.sampleRate << " Hz"
              << "\nBit per campione: " << header.bitsPerSample << "\n";

    int bytesPerSample = header.bitsPerSample / 8; //byte for audio sample -> 16bit/2byte for every sample
    int numSamples = header.subchunk2Size / bytesPerSample; //audio sample number 

    // Alloca memoria allineata a 16 byte per i campioni
    int16_t* alignedData;
    if (posix_memalign(reinterpret_cast<void**>(&alignedData), SSE_DATA_LANE, numSamples * sizeof(int16_t)) != 0) {
        std::cerr << "Errore: allocazione memoria fallita." << std::endl;
        return;
    }
    if (!alignedData) {
        std::cerr << "Errore: allocazione memoria fallita." << std::endl;
        return;
    }
    memset(alignedData, 0, numSamples * sizeof(int16_t));

    // Leggi i campioni direttamente nell'array allineato
    file.read(reinterpret_cast<char*>(alignedData), numSamples * sizeof(int16_t));
    __m128i *PuntatoreAllaMusica = (__m128i*)alignedData;
    __m128i passo;
    std::cout << "Valori dei campioni (SIMD):\n";
    for (int i = 0; i < numSamples/8; ++i) {
         passo = _mm_load_si128(PuntatoreAllaMusica+i);
         //print_register(passo);
    }

    // Stampa sequenziale dei campioni per confronto
    std::cout << "Valori dei campioni (sequenziale):\n";
    for (int i = 0; i < numSamples; ++i) {
        std::cout << alignedData[i] << " ";
        if ((i + 1) % 8 == 0) { // Stampa 8 valori per riga
            std::cout << "\n";
        }
    }
    std::cout << std::endl;

    free(alignedData);
    file.close();
}

int main() {
    printSampleValues("/home/tomas/Desktop/samples/124bpm_drums.wav");
    return 0;
}


/*
#include <iostream>
#include <fstream>
#include <vector>

struct WAVHeader {
    char chunkID[4];
    uint32_t chunkSize;
    char format[4];
    char subchunk1ID[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char subchunk2ID[4];
    uint32_t subchunk2Size;
};

void printSampleValues(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Errore: impossibile aprire il file." << std::endl;
        return;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    std::cout << "Numero di canali: " << header.numChannels
              << "\nFrequenza di campionamento: " << header.sampleRate << " Hz"
              << "\nBit per campione: " << header.bitsPerSample << "\n";

    int bytesPerSample = header.bitsPerSample / 8;
    int numSamples = header.subchunk2Size / bytesPerSample;
    std::vector<char> sampleBuffer(bytesPerSample);

    std::cout << "Valori dei campioni:\n";
    for (int i = 0; i < numSamples; ++i) {
        file.read(sampleBuffer.data(), bytesPerSample);

        int sampleValue = 0;
        switch (header.bitsPerSample) {
            case 16:
                sampleValue = *reinterpret_cast<int16_t*>(sampleBuffer.data());
                break;
            case 8:
                sampleValue = *reinterpret_cast<int8_t*>(sampleBuffer.data());
                break;
            default:
                std::cerr << "Formato dei campioni non supportato.\n";
                return;
        }

        std::cout << "Campione " << i + 1 << ": " << sampleValue << '\n';
    }
}

int main() {
    printSampleValues("/home/tomas/Desktop/samples/124bpm_drums.wav");
    return 0;
}
*/
