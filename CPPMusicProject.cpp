#include <iostream>
#include <fstream>
#include <vector>

struct WAVHeader {
    char chunkID[4]; //type of file
    uint32_t chunkSize; //file size 
    char format[4]; //format file, our case we have WAVE for file WAV
    char subchunk1ID[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels; //channel number
    uint32_t sampleRate; //Hz 44100
    uint32_t byteRate; //number of byte every audio second
    uint16_t blockAlign; //number of byte for audio sample
    uint16_t bitsPerSample;
    char subchunk2ID[4];
    uint32_t subchunk2Size;
};
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
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Errore: impossibile aprire il file." << std::endl;
        return;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    printWAVHeader(header);
    std::cout << "\nNumero di canali: " << header.numChannels
              << "\nFrequenza di campionamento: " << header.sampleRate << " Hz"
              << "\nBit per campione: " << header.bitsPerSample << "\n";

    int bytesPerSample = header.bitsPerSample / 8;
    int numSamples = header.subchunk2Size / bytesPerSample;
    std::vector<char> sampleBuffer(bytesPerSample);

    std::cout << "Valori dei campioni:\n";
    for (int i = 0; i < numSamples; ++i) {
        file.read(sampleBuffer.data(), bytesPerSample);
        int sampleValue = *reinterpret_cast<int16_t*>(sampleBuffer.data());
        //std::cout << "Campione " << i + 1 << ": " << sampleValue << '\n';
    }
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
