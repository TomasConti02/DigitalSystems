 //g++ -o wav_reader MusicProject.cpp -msse2 -O3
#include <iostream>
#include <fstream>
#include <cstdlib> // per aligned_alloc o posix_memalign
#include <cstring> // per memset
#include <immintrin.h> // Intrinseci di Intel x86
#define VECTOR_LENGTH 8
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
void applyLowPassFilterSIMD(__m128i *PuntatoreAllaMusica, int numSamples, float alpha) {
    __m128i prevSample = _mm_setzero_si128();  // Inizializza il campione precedente a zero
    __m128i alphaVec = _mm_set1_epi16(static_cast<int>(alpha * 32768));  // Moltiplicatore alpha (32768 per la conversione a 16 bit)
    __m128i oneMinusAlphaVec = _mm_set1_epi16(static_cast<int>((1.0f - alpha) * 32768));  // (1 - alpha)
    
    // Processa i campioni in blocchi di VECTOR_LENGTH
    for (int i = 0; i < numSamples / VECTOR_LENGTH; ++i) {
        // Carica i campioni correnti nel registro
        __m128i currentSample = _mm_load_si128(&PuntatoreAllaMusica[i]);

        // Applica il filtro passa basso in parallelo (y[n] = alpha * x[n] + (1 - alpha) * y[n-1])
        // 1. Moltiplica currentSample per alpha
        __m128i currentSampleScaled = _mm_mullo_epi16(currentSample, alphaVec);

        // 2. Moltiplica prevSample per (1 - alpha)
        __m128i prevSampleScaled = _mm_mullo_epi16(prevSample, oneMinusAlphaVec);

        // 3. Somma i due risultati
        __m128i result = _mm_add_epi16(currentSampleScaled, prevSampleScaled);

        // 4. Memorizza il risultato nel registro
        prevSample = result;  // Aggiorna il campione precedente

        // Stampa il risultato (opzionale, per debug)
        print_register(result);  // Visualizza il risultato
    }
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
    int numSamples = header.subchunk2Size / bytesPerSample; //audio sample number, NUMERO CAMPIONI

    // Alloca memoria allineata a 16 byte per i campioni
    int16_t* alignedData;
    if (posix_memalign(reinterpret_cast<void**>(&alignedData), SSE_DATA_LANE, numSamples * sizeof(int16_t)) != 0) { //UN CAMPIONE 16byte
        std::cerr << "Errore: allocazione memoria fallita." << std::endl;
        return;
    }
    if (!alignedData) {
        std::cerr << "Errore: allocazione memoria fallita." << std::endl;
        return;
    }
   // memset(alignedData, 0, numSamples * sizeof(int16_t));
    // Leggi i campioni direttamente nell'array allineato
    file.read(reinterpret_cast<char*>(alignedData), numSamples * sizeof(int16_t));
    __m128i *PuntatoreAllaMusica = (__m128i*)alignedData;
    __m128i passo;
    std::cout << "Valori dei campioni (SIMD):\n";
    std::cout << "Il resto della divisione di " << numSamples << " diviso " << VECTOR_LENGTH << " è: " << numSamples % VECTOR_LENGTH << std::endl;
    //STAMPA PARALLELA
    
    int i=0;
    if(numSamples % VECTOR_LENGTH>0){
         for (int i = 0; i <((numSamples/VECTOR_LENGTH)+1); ++i) {
             passo = _mm_load_si128(PuntatoreAllaMusica+i);
            print_register(passo);
         }
    }else{
         for (int i = 0; i < numSamples/VECTOR_LENGTH; ++i) {
            passo = _mm_load_si128(PuntatoreAllaMusica+i);
            print_register(passo);
        }
    }
    /*
    __m128i prevSample = _mm_setzero_si128();  // Inizializza a zero
    __m128i smoothingFactor = _mm_set1_epi16(128);  // Fattore di smussatura
    for (int i = 0; i < numSamples / VECTOR_LENGTH; ++i) {
        __m128i currentSample = _mm_load_si128(&PuntatoreAllaMusica[i]);
        __m128i result = _mm_adds_epu8(currentSample, prevSample);  // Somma dei campioni
        result = _mm_srli_epi16(result, 1);  // Operazione di media (shift destra di 1 bit)
        prevSample = result;  // Aggiorna il campione precedente
        print_register(result);
    }
    */
    /*
    // Stampa SEQUENZIALE dei campioni per confronto
    std::cout << "Valori dei campioni (sequenziale):\n";
    for ( i = 0; i < numSamples; ++i) {
        std::cout << alignedData[i] << " ";
        if ((i + 1) % 8 == 0) { // Stampa 8 valori per riga
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
    */
    //applyLowPassFilterSIMD(PuntatoreAllaMusica,numSamples, 0.1 );
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
/*#include <iostream>
#include <fstream>
#include <vector>
#include <fftw3.h>

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

void printWAVHeader(const WAVHeader& header) {
    std::cout << "sampleRate: " << header.sampleRate << " Hz" << std::endl;
    std::cout << "bitsPerSample: " << header.bitsPerSample << std::endl;
    std::cout << "numChannels: " << header.numChannels << std::endl;
}

void applyFFT(const std::vector<int16_t>& audioData) {
    int N = audioData.size();

    // Creazione dei vettori per il risultato FFT
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan p = fftw_plan_dft_r2c_1d(N, const_cast<int16_t*>(audioData.data()), out, FFTW_ESTIMATE);

    // Esegui FFT
    fftw_execute(p);

    // Stampa il risultato della FFT (frequenza, ampiezza)
    for (int i = 0; i < N / 2; i++) { // Solo metà delle frequenze
        double real = out[i][0];
        double imag = out[i][1];
        double amplitude = sqrt(real * real + imag * imag);
        std::cout << "Frequenza: " << i * 44100.0 / N << " Hz, Ampiezza: " << amplitude << std::endl;
    }

    // Libera la memoria
    fftw_destroy_plan(p);
    fftw_free(out);
}

void readWAVFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Errore nell'aprire il file WAV." << std::endl;
        return;
    }

    // Leggi l'intestazione WAV
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    printWAVHeader(header);

    // Calcola il numero di campioni
    int bytesPerSample = header.bitsPerSample / 8;
    int numSamples = header.subchunk2Size / bytesPerSample;

    // Leggi i campioni audio in un array
    std::vector<int16_t> audioData(numSamples);
    file.read(reinterpret_cast<char*>(audioData.data()), numSamples * sizeof(int16_t));

    file.close();

    // Applica la FFT sul segnale audio
    applyFFT(audioData);
}

int main() {
    // Specifica il percorso del file WAV
    std::string filename = "/path/to/your/audio.wav";
    readWAVFile(filename);
    return 0;
}
*/
